import json
import base64
import uuid
from django.core.files.base import ContentFile # To save strings as files
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import ExamSession, ViolationLog

class ProctoringConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        await self.get_or_create_session()
        await self.accept()

    async def disconnect(self, close_code):
        await self.close_session()

    async def receive(self, text_data):
        data = json.loads(text_data)
        
        if data.get('type') == 'metadata':
            violations = data.get('violations', [])
            pose = data.get('pose', {})
            image_data = data.get('image', None) # <--- Get the Base64 String

            if violations:
                await self.save_violations(violations, pose, image_data)

    # --- DB METHODS ---

    @database_sync_to_async
    def get_or_create_session(self):
        user = self.scope.get("user")
        if not user or not user.is_authenticated:
            user = None
        session, _ = ExamSession.objects.get_or_create(
            session_id=self.session_id,
            defaults={'student': user}
        )
        self.db_session = session

    @database_sync_to_async
    def save_violations(self, violations_list, pose, image_base64):
        # 1. Process Image if it exists
        image_file = None
        if image_base64:
            try:
                # Base64 format: "data:image/jpeg;base64,/9j/4AAQSw..."
                # Split at the comma to get the raw data
                format, imgstr = image_base64.split(';base64,') 
                ext = format.split('/')[-1] # e.g., "jpeg"
                
                # Create a random filename
                filename = f"{uuid.uuid4()}.{ext}"
                
                # Convert to Django ContentFile
                image_file = ContentFile(base64.b64decode(imgstr), name=filename)
            except Exception as e:
                print(f"Error decoding image: {e}")

        # 2. Save Logs
        # We save the image ONLY to the first violation in the list 
        # to save space (since they happen at the exact same millisecond)
        
        for i, v_text in enumerate(violations_list):
            ViolationLog.objects.create(
                session=self.db_session,
                violation_type=v_text,
                pitch=pose.get('pitch', 0),
                yaw=pose.get('yaw', 0),
                roll=pose.get('roll', 0),
                snapshot=image_file if i == 0 else None # Attach image to first log only
            )
            
        self.db_session.total_violations += len(violations_list)
        self.db_session.save()

    @database_sync_to_async
    def close_session(self):
        if hasattr(self, 'db_session'):
            self.db_session.is_active = False
            self.db_session.save()