from fastapi import FastAPI, APIRouter, Header
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# ============ MODELS ============

# Profile Models
class Profile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    firebase_uid: Optional[str] = None  # Firebase User ID for linking profiles
    name: str
    study_field: Optional[str] = None  # "Sayısal" / "EA" / "Sözel"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProfileCreate(BaseModel):
    firebase_uid: Optional[str] = None
    name: str
    study_field: Optional[str] = None

# Program Models
class ProgramTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lesson: str
    topic: str
    duration: int  # minutes
    completed: bool = False
    day: str  # "Pazartesi", "Salı", etc.

class Program(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    exam_goal: str  # "TYT" / "AYT" / "TYT + AYT"
    daily_hours: str  # "1-2" / "2-4" / "4+"
    study_days: int  # 1-7
    tasks: List[ProgramTask] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProgramCreate(BaseModel):
    profile_id: str
    exam_goal: str
    daily_hours: str
    study_days: int

class ProgramUpdate(BaseModel):
    exam_goal: Optional[str] = None
    daily_hours: Optional[str] = None
    study_days: Optional[int] = None
    tasks: Optional[List[ProgramTask]] = None

# Room Models
class RoomParticipant(BaseModel):
    id: str
    name: str
    study_field: Optional[str] = None

class TimerState(BaseModel):
    is_running: bool = False
    duration_minutes: int = 25
    remaining_seconds: int = 0
    started_at: Optional[str] = None

class Room(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    code: str = Field(default_factory=lambda: str(uuid.uuid4())[:6].upper())
    owner_id: str
    participants: List[RoomParticipant] = []
    timer_state: TimerState = Field(default_factory=TimerState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RoomCreate(BaseModel):
    name: str
    owner_name: str
    owner_study_field: Optional[str] = None

class RoomJoin(BaseModel):
    room_code: str
    user_name: str
    user_study_field: Optional[str] = None

# Message Models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    room_id: str
    user_id: str
    user_name: str
    user_study_field: Optional[str] = None
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MessageCreate(BaseModel):
    room_id: str
    user_id: str
    user_name: str
    user_study_field: Optional[str] = None
    content: str

# Old Models (keep for compatibility)
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# Exam Result Models (Net Takibi)
class ExamResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    firebase_uid: str
    exam_type: str  # "TYT" / "AYT"
    date: str  # YYYY-MM-DD format
    net_score: float
    exam_name: Optional[str] = None  # Deneme adı (opsiyonel)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ExamResultCreate(BaseModel):
    exam_type: str
    date: str
    net_score: float
    exam_name: Optional[str] = None

# ============ API ROUTES ============

# Health Check
@api_router.get("/")
async def root():
    return {"message": "İzlek API"}

# Old endpoints (keep for compatibility)
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# ============ PROFILE ENDPOINTS ============

@api_router.post("/profiles", response_model=Profile)
async def create_profile(input: ProfileCreate):
    if not input.name or input.name.strip() == "":
        return {"error": "İsim boş olamaz"}
    
    profile_obj = Profile(**input.model_dump())
    doc = profile_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.profiles.insert_one(doc)
    return profile_obj

@api_router.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str):
    profile = await db.profiles.find_one({"id": profile_id}, {"_id": 0})
    if profile:
        if isinstance(profile.get('created_at'), str):
            profile['created_at'] = datetime.fromisoformat(profile['created_at'])
    return profile

@api_router.get("/profiles/by-firebase-uid/{firebase_uid}", response_model=Profile)
async def get_profile_by_firebase_uid(firebase_uid: str):
    """Get user profile by Firebase UID - used for checking if onboarding is completed"""
    profile = await db.profiles.find_one({"firebase_uid": firebase_uid}, {"_id": 0})
    if profile:
        if isinstance(profile.get('created_at'), str):
            profile['created_at'] = datetime.fromisoformat(profile['created_at'])
    return profile

# ============ PROGRAM ENDPOINTS ============

@api_router.post("/programs", response_model=Program)
async def create_program(input: ProgramCreate):
    program_obj = Program(**input.model_dump())
    
    # Auto-generate simple starter tasks
    program_obj.tasks = generate_starter_tasks(input.exam_goal, input.daily_hours, input.study_days)
    
    doc = program_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.programs.insert_one(doc)
    return program_obj

@api_router.get("/programs/{profile_id}", response_model=List[Program])
async def get_programs(profile_id: str):
    programs = await db.programs.find({"profile_id": profile_id}, {"_id": 0}).to_list(1000)
    
    for program in programs:
        if isinstance(program.get('created_at'), str):
            program['created_at'] = datetime.fromisoformat(program['created_at'])
    
    return programs

@api_router.put("/programs/{program_id}", response_model=Program)
async def update_program(program_id: str, input: ProgramUpdate):
    update_data = {k: v for k, v in input.model_dump().items() if v is not None}
    
    if update_data:
        await db.programs.update_one(
            {"id": program_id},
            {"$set": update_data}
        )
    
    program = await db.programs.find_one({"id": program_id}, {"_id": 0})
    if program and isinstance(program.get('created_at'), str):
        program['created_at'] = datetime.fromisoformat(program['created_at'])
    
    return program

@api_router.delete("/programs/{program_id}")
async def delete_program(program_id: str):
    result = await db.programs.delete_one({"id": program_id})
    return {"deleted": result.deleted_count > 0}

# ============ ROOM ENDPOINTS ============

@api_router.post("/rooms", response_model=Room)
async def create_room(input: RoomCreate):
    if not input.name or input.name.strip() == "":
        return {"error": "Oda adı boş olamaz"}
    if not input.owner_name or input.owner_name.strip() == "":
        return {"error": "İsim boş olamaz"}
    
    room_obj = Room(
        name=input.name,
        owner_id=str(uuid.uuid4())
    )
    
    # Add owner as first participant
    owner = RoomParticipant(
        id=room_obj.owner_id,
        name=input.owner_name,
        study_field=input.owner_study_field
    )
    room_obj.participants.append(owner)
    
    doc = room_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.rooms.insert_one(doc)
    return room_obj

@api_router.post("/rooms/join", response_model=Room)
async def join_room(input: RoomJoin):
    if not input.user_name or input.user_name.strip() == "":
        return {"error": "İsim boş olamaz"}
    if not input.room_code or input.room_code.strip() == "":
        return {"error": "Oda kodu boş olamaz"}
    
    room = await db.rooms.find_one({"code": input.room_code.upper()}, {"_id": 0})
    
    if not room:
        return {"error": "Oda bulunamadı"}
    
    # Add new participant
    new_participant = RoomParticipant(
        id=str(uuid.uuid4()),
        name=input.user_name,
        study_field=input.user_study_field
    )
    
    await db.rooms.update_one(
        {"code": input.room_code.upper()},
        {"$push": {"participants": new_participant.model_dump()}}
    )
    
    updated_room = await db.rooms.find_one({"code": input.room_code.upper()}, {"_id": 0})
    if updated_room and isinstance(updated_room.get('created_at'), str):
        updated_room['created_at'] = datetime.fromisoformat(updated_room['created_at'])
    
    return updated_room

@api_router.get("/rooms/{room_id}", response_model=Room)
async def get_room(room_id: str):
    room = await db.rooms.find_one({"id": room_id}, {"_id": 0})
    if room and isinstance(room.get('created_at'), str):
        room['created_at'] = datetime.fromisoformat(room['created_at'])
    return room

@api_router.get("/rooms/code/{room_code}", response_model=Room)
async def get_room_by_code(room_code: str):
    room = await db.rooms.find_one({"code": room_code.upper()}, {"_id": 0})
    if room and isinstance(room.get('created_at'), str):
        room['created_at'] = datetime.fromisoformat(room['created_at'])
    return room

@api_router.put("/rooms/{room_id}/timer")
async def update_timer(room_id: str, timer_state: TimerState):
    await db.rooms.update_one(
        {"id": room_id},
        {"$set": {"timer_state": timer_state.model_dump()}}
    )
    return {"success": True}

# ============ MESSAGE ENDPOINTS ============

@api_router.post("/messages", response_model=Message)
async def create_message(input: MessageCreate):
    if not input.content or input.content.strip() == "":
        return {"error": "Mesaj boş olamaz"}
    
    message_obj = Message(**input.model_dump())
    doc = message_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    await db.messages.insert_one(doc)
    return message_obj

@api_router.get("/messages/{room_id}", response_model=List[Message])
async def get_messages(room_id: str):
    messages = await db.messages.find({"room_id": room_id}, {"_id": 0}).sort("timestamp", 1).to_list(1000)
    
    for message in messages:
        if isinstance(message.get('timestamp'), str):
            message['timestamp'] = datetime.fromisoformat(message['timestamp'])
    
    return messages

# ============ EXAM RESULT ENDPOINTS (Net Takibi) ============

@api_router.post("/exams", response_model=ExamResult)
async def create_exam_result(input: ExamResultCreate, firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    """Create a new exam result (net) for the authenticated user"""
    if not input.exam_type or input.exam_type not in ["TYT", "AYT"]:
        return {"error": "Geçersiz sınav türü (TYT/AYT olmalı)"}
    
    if not input.date:
        return {"error": "Tarih boş olamaz"}
    
    if input.net_score is None or input.net_score < 0:
        return {"error": "Net skoru geçersiz"}
    
    exam_obj = ExamResult(
        firebase_uid=firebase_uid,
        exam_type=input.exam_type,
        date=input.date,
        net_score=input.net_score,
        exam_name=input.exam_name
    )
    
    doc = exam_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.exam_results.insert_one(doc)
    return exam_obj

@api_router.get("/exams", response_model=List[ExamResult])
async def get_exam_results(firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    """Get all exam results for the authenticated user"""
    results = await db.exam_results.find(
        {"firebase_uid": firebase_uid}, 
        {"_id": 0}
    ).sort("date", -1).to_list(1000)
    
    for result in results:
        if isinstance(result.get('created_at'), str):
            result['created_at'] = datetime.fromisoformat(result['created_at'])
    
    return results


# ============ HELPER FUNCTIONS ============

def generate_starter_tasks(exam_goal: str, daily_hours: str, study_days: int) -> List[ProgramTask]:
    """Generate simple starter program tasks based on user preferences"""
    
    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"][:study_days]
    
    # Simple task templates based on exam goal
    task_templates = {
        "TYT": [
            {"lesson": "Matematik", "topic": "Temel Kavramlar", "duration": 45},
            {"lesson": "Türkçe", "topic": "Sözcük - Cümle", "duration": 30},
            {"lesson": "Fen", "topic": "Fizik - Hareket", "duration": 40},
            {"lesson": "Sosyal", "topic": "Tarih - İnkılap", "duration": 35},
        ],
        "AYT": [
            {"lesson": "Matematik", "topic": "İntegral", "duration": 50},
            {"lesson": "Fizik", "topic": "Elektrik", "duration": 45},
            {"lesson": "Kimya", "topic": "Organik", "duration": 40},
            {"lesson": "Biyoloji", "topic": "Hücre", "duration": 35},
        ],
        "TYT + AYT": [
            {"lesson": "TYT Matematik", "topic": "Temel Matematik", "duration": 40},
            {"lesson": "AYT Matematik", "topic": "İleri Matematik", "duration": 40},
            {"lesson": "Türkçe", "topic": "Okuma - Anlama", "duration": 30},
            {"lesson": "Fen", "topic": "Fizik & Kimya", "duration": 35},
        ]
    }
    
    templates = task_templates.get(exam_goal, task_templates["TYT"])
    tasks = []
    
    for day in days:
        for template in templates[:2]:  # 2 task per day for simplicity
            tasks.append(ProgramTask(
                lesson=template["lesson"],
                topic=template["topic"],
                duration=template["duration"],
                day=day,
                completed=False
            ))
    
    return tasks

# Include the router in the main app
app.include_router(api_router)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()