from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.chatbot.chatbot import Chatbot
import random

# TODO: titulo ajustar quando tiver nome, colocar no readme também, no package, manifest, env do site
server = FastAPI(title="Transbot")


@server.websocket("/talk", 'Talk')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    chatbot = Chatbot()
    chatbot.load()
    client_id = random.randint(1, 1000000)
    print('New client connected. ID:', client_id)
    while True:
        try:
            body = await websocket.receive_json()
            result = chatbot.execute(body['question'], body['lastAnswer'])
            print({"result": result, "body": body})
            await websocket.send_json({"status": 200, "answer": result.strip()})
        except WebSocketDisconnect:
            print('Client disconnected. ID:', client_id)
            break
        except Exception as e:
            print('Client disconnected becouse error. ID:', client_id)
            print('error:', e)
            break
