#!/bin/bash
uvicorn --host 0.0.0.0 --port 8086 models.demos.yolov4.web_demo.server.fast_api_yolov4:app
