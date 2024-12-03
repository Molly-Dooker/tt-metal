// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_memory.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tt_elffile.hpp"
#include "tt_metal/common/assert.hpp"

namespace ll_api {

memory::memory() {
    constexpr uint32_t initial_data_space_ = 0x400;
    constexpr uint32_t initial_span_space_ = 4;

    data_.reserve(initial_data_space_);
    link_spans_.reserve(initial_span_space_);
}

memory::memory(std::string const& path, Packing pack_type, Relocate relo_type) {
    ElfFile elf;

    elf.ReadImage(path);
    if (relo_type == Relocate::XIP) {
        elf.MakeExecuteInPlace();
    }

    // The ELF file puts the text segment first, but memory wants
    // ordered spans.
    // FIXME: Perhaps we can relax that?
    auto emit_segment = [&](ElfFile::Segment const& segment) {
        TT_ASSERT(segment.relocs.empty(), "Unexpected dynamic relocations");
        link_spans_.emplace_back(segment.address, segment.contents.size());
        data_.insert(data_.end(), segment.contents.begin(), segment.contents.end());
    };
    auto* text = &elf.GetSegments()[0];
    for (auto& segment : std::span(elf.GetSegments()).subspan(1)) {
        if (text && segment.address > text->address) {
            emit_segment(*text);
            text = nullptr;
        }
        emit_segment(segment);
    }
    if (text) {
        emit_segment(*text);
    }

    set_text_size(elf.GetSegments()[0].contents.size() * sizeof(word_t));
    set_packed_size(data_.size() * sizeof(uint32_t));
}

bool memory::operator==(const memory& other) const { return data_ == other.data_ && link_spans_ == other.link_spans_; }

void memory::fill_from_mem_template(
    const memory& mem_template,
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    link_spans_ = mem_template.link_spans_;
    data_.resize(mem_template.data_.size());
    process_spans(callback);
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::const_iterator cit = data_.cbegin() + offset;
        callback(cit, span.addr, span.len);
        offset += span.len;
    }
}

void memory::process_spans(
    const std::function<void(std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback) {
    uint32_t offset = 0;
    for (const auto& span : link_spans_) {
        std::vector<uint32_t>::iterator it = data_.begin() + offset;
        callback(it, span.addr, span.len);
        offset += span.len;
    }
}

// Takes spans and merges the data to the text span
// Used for kernels (not firmware)
// Spans get packed for kernels so they can be loaded in one NOC transaction
// A symbol at the end of the text segment allows the FW to find the data segment to copy into place
void memory::pack_data_into_text(std::uint64_t text_start, std::uint64_t data_start) {
    uint64_t text_end, data_end;
    if (text_start > data_start) {
        text_end = std::numeric_limits<uint64_t>::max();
        data_end = text_start;
    } else {
        text_end = data_start;
        data_end = std::numeric_limits<uint64_t>::max();
    }

    TT_ASSERT(this->link_spans_.size() != 0);
    TT_ASSERT(link_spans_.size() <= 2);

    std::vector<word_t> new_data;

    bool text_is_second =
        link_spans_.size() == 2 && link_spans_[0].addr >= data_start && link_spans_[0].addr < data_end;
    auto const& text = link_spans_[text_is_second];
    TT_ASSERT(text.addr >= text_start && text.addr < text_end);

    new_data.resize(this->data_.size());
    span new_span{text.addr, text.len};

    size_t offset = text_is_second ? link_spans_[0].len : 0;
    new_data.insert(new_data.end(), &data_[offset], &data_[offset] + text.len);

    if (link_spans_.size() == 2) {
        offset = text_is_second ? 0 : text.len;
        auto const& data = link_spans_[!text_is_second];
        TT_ASSERT(data.addr >= data_start && data.addr < data_end);
        new_span.len += data.len;
        new_data.insert(new_data.end(), &data_[offset], &data_[offset] + data.len);
    }

    this->link_spans_.resize(1);
    this->link_spans_[0] = new_span;
    this->data_ = std::move(new_data);
    this->text_addr_ = new_span.addr;
}

}  // namespace ll_api
