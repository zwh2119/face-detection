import os
import random
import shutil
import time
import contextlib
import threading
import asyncio

import base64

import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from car_detection import CarDetection
import field_codec_utils


class ServiceServer:

    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute('/predict',
                     self.cal,
                     response_class=JSONResponse,
                     methods=['POST']

                     ),
        ], log_level='trace', timeout=6000)

        self.estimator = CarDetection({
            'weights': 'yolov5s.pt',
            # 'device': 'cpu'
            'device': 'cuda:0'
        })

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

    async def cal(self, file: UploadFile = File(...), data: str = Form(...)):

        tmp_path = f'tmp_receive_{time.time()}.mp4'
        with open(tmp_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
            del file

        content = []
        video_cap = cv2.VideoCapture(tmp_path)
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            content.append(frame)
        os.remove(tmp_path)
        start = time.time()
        result = await self.estimator(content)
        end = time.time()
        print(f'process time:{end-start}s')
        assert type(result) is dict

        return result


app_server = ServiceServer()
app = app_server.app
