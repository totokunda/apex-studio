from fastapi import APIRouter, WebSocketDisconnect
from fastapi.websockets import WebSocket
from .ws_manager import websocket_manager

router = APIRouter(prefix="/ws")


@router.websocket("/engine")
async def engine_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")


@router.websocket("/job/{job_id}")
async def job_status_websocket(websocket: WebSocket, job_id: str):
    """Websocket endpoint for receiving job status updates"""
    await websocket_manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            # Client can send ping/pong messages to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, job_id)
