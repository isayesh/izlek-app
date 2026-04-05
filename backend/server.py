from fastapi import FastAPI, APIRouter, Header, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import re
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timedelta, timezone
from pymongo.errors import DuplicateKeyError
import bcrypt


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

cors_origins = os.environ["CORS_ORIGINS"].split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


HANDLE_PATTERN = re.compile(r"^[a-z0-9_]{3,20}$")
HANDLE_VALIDATION_MESSAGE = "Handle 3-20 karakter olmalı ve yalnızca küçük harf, rakam, underscore içermelidir."
HANDLE_DUPLICATE_MESSAGE = "Bu handle zaten kullanımda."
FRIEND_REQUEST_PENDING = "pending"
FRIEND_REQUEST_ACCEPTED = "accepted"
FRIEND_REQUEST_REJECTED = "rejected"
FRIEND_RELATIONSHIP_NONE = "none"
FRIEND_RELATIONSHIP_PENDING_OUTGOING = "outgoing_pending"
FRIEND_RELATIONSHIP_PENDING_INCOMING = "incoming_pending"
FRIEND_RELATIONSHIP_FRIENDS = "friends"
ROOM_TYPE_PUBLIC = "public"
ROOM_TYPE_PRIVATE = "private"
ROOM_PASSWORD_MIN_LENGTH = 6
ROOM_PASSWORD_REQUIRED_MESSAGE = "Bu oda özel. Devam etmek için oda şifresini gir."
ROOM_PASSWORD_INVALID_MESSAGE = "Oda şifresi hatalı."
ROOM_PASSWORD_VALIDATION_MESSAGE = f"Oda şifresi en az {ROOM_PASSWORD_MIN_LENGTH} karakter olmalıdır."


# ============ MODELS ============

# Profile Models
class Profile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    firebase_uid: Optional[str] = None  # Firebase User ID for linking profiles
    name: str
    username: Optional[str] = None
    handle: Optional[str] = None
    email: Optional[str] = None
    study_goal: Optional[str] = None
    daily_study_hours: Optional[float] = None
    avatar_url: Optional[str] = None
    study_field: Optional[str] = None  # "Sayısal" / "EA" / "Sözel"
    streak_count: int = 0
    last_active_date: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProfileCreate(BaseModel):
    firebase_uid: Optional[str] = None
    name: str
    username: Optional[str] = None
    handle: Optional[str] = None
    email: Optional[str] = None
    study_goal: Optional[str] = None
    daily_study_hours: Optional[float] = None
    avatar_url: Optional[str] = None
    study_field: Optional[str] = None

class SimpleProfilePayload(BaseModel):
    firebase_uid: str
    username: str
    email: str
    handle: Optional[str] = None
    study_goal: Optional[str] = None
    daily_study_hours: Optional[float] = None
    avatar_url: Optional[str] = None


def normalize_existing_handle(raw_handle: Optional[str]) -> Optional[str]:
    if raw_handle is None:
        return None

    normalized = str(raw_handle).strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]

    if normalized == "":
        return None

    return normalized if HANDLE_PATTERN.fullmatch(normalized) else None


def normalize_input_handle(raw_handle: Optional[str]) -> Optional[str]:
    if raw_handle is None:
        return None

    normalized = raw_handle.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]

    if normalized == "":
        return None

    if not HANDLE_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=400, detail=HANDLE_VALIDATION_MESSAGE)

    return normalized


def resolve_public_profile_name(profile: Optional[dict]) -> str:
    profile = profile or {}

    handle = normalize_existing_handle(profile.get("handle"))
    if handle:
        return f"@{handle}"

    username = (profile.get("username") or "").strip()
    if username:
        return username

    name = (profile.get("name") or "").strip()
    if name:
        return name

    return "Bilinmeyen Kullanıcı"


def resolve_public_username(profile: Optional[dict]) -> str:
    profile = profile or {}

    username = (profile.get("username") or "").strip()
    if username:
        return username

    name = (profile.get("name") or "").strip()
    if name:
        return name

    return "Bilinmeyen Kullanıcı"


def format_handle_display(handle: Optional[str]) -> Optional[str]:
    normalized_handle = normalize_existing_handle(handle)
    return f"@{normalized_handle}" if normalized_handle else None


async def ensure_handle_is_unique(
    handle: Optional[str],
    *,
    firebase_uid: Optional[str] = None,
    profile_id: Optional[str] = None,
):
    if not handle:
        return

    query = {"handle": handle}
    if firebase_uid is not None:
        query["firebase_uid"] = {"$ne": firebase_uid}
    if profile_id is not None:
        query["id"] = {"$ne": profile_id}

    existing_profile = await db.profiles.find_one(query, {"_id": 0, "id": 1})
    if existing_profile:
        raise HTTPException(status_code=409, detail=HANDLE_DUPLICATE_MESSAGE)


def format_profile_document(profile: Optional[dict]) -> Optional[Profile]:
    if not profile:
        return None

    normalized = dict(profile)
    created_at = normalized.get("created_at")

    if isinstance(created_at, str):
        normalized["created_at"] = datetime.fromisoformat(created_at)

    normalized["handle"] = normalize_existing_handle(normalized.get("handle"))
    username = (normalized.get("username") or normalized.get("name") or "").strip()
    normalized["username"] = username
    normalized["name"] = (normalized.get("name") or username).strip()
    normalized["email"] = (normalized.get("email") or None)
    normalized["study_goal"] = (normalized.get("study_goal") or None)
    normalized["avatar_url"] = (normalized.get("avatar_url") or None)
    normalized["streak_count"] = int(normalized.get("streak_count") or 0)
    normalized["last_active_date"] = normalized.get("last_active_date") or None

    return Profile(**normalized)


def build_profile_document(input_data: SimpleProfilePayload, existing_profile: Optional[dict] = None) -> dict:
    existing_profile = existing_profile or {}

    username = input_data.username.strip()
    email = input_data.email.strip()
    input_handle = input_data.handle if input_data.handle is not None else existing_profile.get("handle")
    handle = normalize_input_handle(input_handle)

    if username == "":
        raise HTTPException(status_code=400, detail="Kullanıcı adı boş olamaz")

    if email == "":
        raise HTTPException(status_code=400, detail="E-posta boş olamaz")

    created_at = existing_profile.get("created_at") or datetime.now(timezone.utc)
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    study_goal = input_data.study_goal.strip() if input_data.study_goal else None
    avatar_url = input_data.avatar_url.strip() if input_data.avatar_url else None

    return {
        "id": existing_profile.get("id", str(uuid.uuid4())),
        "firebase_uid": input_data.firebase_uid,
        "name": username,
        "username": username,
        "handle": handle,
        "email": email,
        "study_goal": study_goal,
        "daily_study_hours": input_data.daily_study_hours,
        "avatar_url": avatar_url,
        "study_field": existing_profile.get("study_field"),
        "streak_count": int(existing_profile.get("streak_count") or 0),
        "last_active_date": existing_profile.get("last_active_date") or None,
        "created_at": created_at,
    }


def get_today_date_string() -> str:
    return datetime.now().date().isoformat()


async def update_profile_streak(firebase_uid: Optional[str], activity_date: Optional[str] = None):
    if not firebase_uid:
        return None

    profile = await db.profiles.find_one(
        {"firebase_uid": firebase_uid},
        {"_id": 0, "streak_count": 1, "last_active_date": 1},
    )
    if not profile:
        return None

    resolved_activity_date = activity_date or get_today_date_string()
    yesterday_date = (datetime.fromisoformat(resolved_activity_date).date() - timedelta(days=1)).isoformat()
    last_active_date = (profile.get("last_active_date") or "").strip()
    current_streak = int(profile.get("streak_count") or 0)

    if last_active_date == resolved_activity_date:
        next_streak = current_streak
    elif last_active_date == yesterday_date:
        next_streak = current_streak + 1 if current_streak > 0 else 1
    else:
        next_streak = 1

    await db.profiles.update_one(
        {"firebase_uid": firebase_uid},
        {"$set": {"streak_count": next_streak, "last_active_date": resolved_activity_date}},
    )
    return {"streak_count": next_streak, "last_active_date": resolved_activity_date}


def build_public_profile_card(profile: dict, relationship_status: str = FRIEND_RELATIONSHIP_NONE) -> "PublicProfileCard":
    handle = normalize_existing_handle(profile.get("handle"))
    return PublicProfileCard(
        profile_id=profile.get("id", ""),
        display_name=resolve_public_profile_name(profile),
        username=resolve_public_username(profile),
        handle=handle,
        handle_display=format_handle_display(handle),
        avatar_url=profile.get("avatar_url") or None,
        study_goal=profile.get("study_goal") or None,
        daily_study_hours=profile.get("daily_study_hours"),
        relationship_status=relationship_status,
    )


def build_friend_request_snapshot(profile: dict) -> dict:
    return {
        "from_username": resolve_public_username(profile),
        "from_handle": normalize_existing_handle(profile.get("handle")),
        "from_avatar_url": profile.get("avatar_url") or None,
    }


async def get_profile_document_by_uid(firebase_uid: str) -> Optional[dict]:
    return await db.profiles.find_one({"firebase_uid": firebase_uid}, {"_id": 0})


async def are_users_friends(user_uid: str, friend_uid: str) -> bool:
    if not user_uid or not friend_uid:
        return False

    friendship_doc = await db.friends.find_one(
        {"user_uid": user_uid, "friend_uid": friend_uid},
        {"_id": 0, "id": 1},
    )
    return friendship_doc is not None


async def get_relationship_status_map(current_uid: str, other_uids: List[str]) -> dict:
    filtered_uids = [uid for uid in other_uids if uid]
    status_map = {uid: FRIEND_RELATIONSHIP_NONE for uid in filtered_uids}

    if not current_uid or not filtered_uids:
        return status_map

    friend_docs = await db.friends.find(
        {"user_uid": current_uid, "friend_uid": {"$in": filtered_uids}},
        {"_id": 0, "friend_uid": 1},
    ).to_list(1000)
    for friend_doc in friend_docs:
        friend_uid = friend_doc.get("friend_uid")
        if friend_uid:
            status_map[friend_uid] = FRIEND_RELATIONSHIP_FRIENDS

    outgoing_request_docs = await db.friend_requests.find(
        {
            "from_uid": current_uid,
            "to_uid": {"$in": filtered_uids},
            "status": FRIEND_REQUEST_PENDING,
        },
        {"_id": 0, "to_uid": 1},
    ).to_list(1000)
    for request_doc in outgoing_request_docs:
        receiver_uid = request_doc.get("to_uid")
        if receiver_uid and status_map.get(receiver_uid) == FRIEND_RELATIONSHIP_NONE:
            status_map[receiver_uid] = FRIEND_RELATIONSHIP_PENDING_OUTGOING

    incoming_request_docs = await db.friend_requests.find(
        {
            "from_uid": {"$in": filtered_uids},
            "to_uid": current_uid,
            "status": FRIEND_REQUEST_PENDING,
        },
        {"_id": 0, "from_uid": 1},
    ).to_list(1000)
    for request_doc in incoming_request_docs:
        sender_uid = request_doc.get("from_uid")
        if sender_uid and status_map.get(sender_uid) == FRIEND_RELATIONSHIP_NONE:
            status_map[sender_uid] = FRIEND_RELATIONSHIP_PENDING_INCOMING

    return status_map


def sort_public_profile_cards(cards: List["PublicProfileCard"], query: str) -> List["PublicProfileCard"]:
    raw_query = query.strip().lower()
    normalized_handle_query = raw_query.lstrip("@")

    def sort_key(card: "PublicProfileCard"):
        handle = (card.handle or "").lower()
        username = (card.username or "").lower()
        display_name = (card.display_name or "").lower()

        if handle and handle == normalized_handle_query:
            priority = 0
        elif username == raw_query:
            priority = 1
        elif handle and handle.startswith(normalized_handle_query):
            priority = 2
        elif username.startswith(raw_query):
            priority = 3
        elif display_name.startswith(raw_query) or display_name.startswith(normalized_handle_query):
            priority = 4
        else:
            priority = 5

        return (priority, username, display_name)

    return sorted(cards, key=sort_key)

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
    avatar_url: Optional[str] = None
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
    room_type: str = ROOM_TYPE_PUBLIC
    is_private: bool = False
    participants: List[RoomParticipant] = []
    timer_state: TimerState = Field(default_factory=TimerState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RoomCreate(BaseModel):
    name: str
    owner_name: str
    owner_id: Optional[str] = None
    owner_avatar_url: Optional[str] = None
    owner_study_field: Optional[str] = None
    room_type: Optional[str] = ROOM_TYPE_PUBLIC
    room_password: Optional[str] = None

class RoomJoin(BaseModel):
    room_code: str
    user_id: Optional[str] = None
    user_name: str
    user_avatar_url: Optional[str] = None
    user_study_field: Optional[str] = None
    room_password: Optional[str] = None

class RoomLeave(BaseModel):
    user_id: str
    user_name: Optional[str] = None

async def create_system_room_message(room_id: str, content: str):
    message_obj = Message(
        room_id=room_id,
        user_id="system",
        user_name="Sistem",
        user_avatar_url=None,
        user_study_field=None,
        content=content,
    )
    doc = message_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.messages.insert_one(doc)


def normalize_room_type(raw_room_type: Optional[str]) -> str:
    normalized = (raw_room_type or ROOM_TYPE_PUBLIC).strip().lower()
    return ROOM_TYPE_PRIVATE if normalized == ROOM_TYPE_PRIVATE else ROOM_TYPE_PUBLIC


def normalize_room_password(raw_password: Optional[str]) -> Optional[str]:
    if raw_password is None:
        return None

    normalized = raw_password.strip()
    return normalized or None


def hash_room_password(raw_password: str) -> str:
    return bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_room_password(raw_password: Optional[str], stored_hash: Optional[str]) -> bool:
    if not stored_hash:
        return True

    normalized_password = normalize_room_password(raw_password)
    if not normalized_password:
        return False

    try:
        return bcrypt.checkpw(normalized_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError:
        return False


def format_room_document(room: Optional[dict]) -> Optional[Room]:
    if not room:
        return None

    normalized = dict(room)
    created_at = normalized.get("created_at")
    if isinstance(created_at, str):
        normalized["created_at"] = datetime.fromisoformat(created_at)

    room_type = normalize_room_type(normalized.get("room_type"))
    normalized["room_type"] = room_type
    normalized["is_private"] = room_type == ROOM_TYPE_PRIVATE or bool(normalized.get("room_password_hash"))
    return Room(**normalized)


# Message Models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    room_id: str
    user_id: str
    user_name: str
    user_avatar_url: Optional[str] = None
    user_study_field: Optional[str] = None
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MessageCreate(BaseModel):
    room_id: str
    user_id: str
    user_name: str
    user_avatar_url: Optional[str] = None
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

# Study Session Models (for timer persistence)
class StudySession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    firebase_uid: str  # User identifier from Firebase
    room_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_saved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accumulated_seconds: int = 0
    is_completed: bool = False

class StudySessionStart(BaseModel):
    firebase_uid: str
    room_id: str

class StudySessionUpdate(BaseModel):
    accumulated_seconds: int

class StudySessionComplete(BaseModel):
    accumulated_seconds: int

# Leaderboard Models
class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    user_name: str
    avatar_url: Optional[str] = None
    total_seconds: int


class PublicProfileCard(BaseModel):
    profile_id: str
    display_name: str
    username: str
    handle: Optional[str] = None
    handle_display: Optional[str] = None
    avatar_url: Optional[str] = None
    study_goal: Optional[str] = None
    daily_study_hours: Optional[float] = None
    relationship_status: Optional[str] = None


class FriendRequestCreate(BaseModel):
    to_profile_id: str


class FriendRequestNotification(BaseModel):
    id: str
    status: str
    created_at: datetime
    from_display_name: str
    from_username: str
    from_handle: Optional[str] = None
    from_handle_display: Optional[str] = None
    from_avatar_url: Optional[str] = None


class FriendActionResponse(BaseModel):
    success: bool
    status: str
    detail: str

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

    normalized_handle = normalize_input_handle(input.handle)
    profile_payload = input.model_dump()
    profile_payload["name"] = input.name.strip()
    profile_payload["username"] = (input.username or input.name).strip()
    profile_payload["handle"] = normalized_handle
    profile_payload["email"] = input.email.strip() if input.email else None
    profile_payload["study_goal"] = input.study_goal.strip() if input.study_goal else None
    profile_payload["avatar_url"] = input.avatar_url.strip() if input.avatar_url else None

    profile_obj = Profile(**profile_payload)
    doc = profile_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()

    await ensure_handle_is_unique(profile_obj.handle, firebase_uid=profile_obj.firebase_uid, profile_id=profile_obj.id)

    try:
        await db.profiles.insert_one(doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail=HANDLE_DUPLICATE_MESSAGE) from exc

    return profile_obj

# Simple profile endpoints used by the profile page
@api_router.post("/profile", response_model=Profile)
async def create_simple_profile(input: SimpleProfilePayload):
    existing_profile = await db.profiles.find_one({"firebase_uid": input.firebase_uid}, {"_id": 0})
    profile_doc = build_profile_document(input, existing_profile)
    stored_doc = {**profile_doc, "created_at": profile_doc["created_at"].isoformat()}

    await ensure_handle_is_unique(
        profile_doc.get("handle"),
        firebase_uid=input.firebase_uid,
        profile_id=profile_doc.get("id"),
    )

    try:
        if existing_profile:
            await db.profiles.update_one({"firebase_uid": input.firebase_uid}, {"$set": stored_doc})
        else:
            await db.profiles.insert_one(stored_doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail=HANDLE_DUPLICATE_MESSAGE) from exc

    return format_profile_document(profile_doc)


@api_router.get("/profile", response_model=Optional[Profile])
async def get_simple_profile(firebase_uid: str = Query(...)):
    profile = await db.profiles.find_one({"firebase_uid": firebase_uid}, {"_id": 0})
    return format_profile_document(profile)


@api_router.put("/profile", response_model=Profile)
async def update_simple_profile(input: SimpleProfilePayload):
    existing_profile = await db.profiles.find_one({"firebase_uid": input.firebase_uid}, {"_id": 0})

    if not existing_profile:
        raise HTTPException(status_code=404, detail="Profil bulunamadı")

    profile_doc = build_profile_document(input, existing_profile)
    stored_doc = {**profile_doc, "created_at": profile_doc["created_at"].isoformat()}

    await ensure_handle_is_unique(
        profile_doc.get("handle"),
        firebase_uid=input.firebase_uid,
        profile_id=profile_doc.get("id"),
    )

    try:
        await db.profiles.update_one({"firebase_uid": input.firebase_uid}, {"$set": stored_doc})
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail=HANDLE_DUPLICATE_MESSAGE) from exc

    return format_profile_document(profile_doc)

@api_router.get("/profiles/{profile_id}", response_model=Optional[Profile])
async def get_profile(profile_id: str):
    profile = await db.profiles.find_one({"id": profile_id}, {"_id": 0})
    return format_profile_document(profile)

@api_router.get("/profiles/by-firebase-uid/{firebase_uid}", response_model=Optional[Profile])
async def get_profile_by_firebase_uid(firebase_uid: str):
    """Get user profile by Firebase UID - used for checking if onboarding is completed"""
    profile = await db.profiles.find_one({"firebase_uid": firebase_uid}, {"_id": 0})
    return format_profile_document(profile)


@api_router.get("/users/search", response_model=List[PublicProfileCard])
async def search_users(
    q: str = Query(..., min_length=1),
    firebase_uid: str = Header(..., alias="X-Firebase-UID"),
):
    search_query = q.strip()
    if not search_query:
        raise HTTPException(status_code=400, detail="Arama sorgusu boş olamaz")

    normalized_handle_query = search_query.lstrip("@")
    profile_docs = await db.profiles.find(
        {
            "firebase_uid": {"$ne": firebase_uid},
            "$or": [
                {"handle": {"$regex": re.escape(normalized_handle_query), "$options": "i"}},
                {"username": {"$regex": re.escape(search_query), "$options": "i"}},
                {"name": {"$regex": re.escape(search_query), "$options": "i"}},
            ],
        },
        {
            "_id": 0,
            "id": 1,
            "firebase_uid": 1,
            "username": 1,
            "name": 1,
            "handle": 1,
            "avatar_url": 1,
            "study_goal": 1,
            "daily_study_hours": 1,
        },
    ).to_list(20)

    if not profile_docs:
        return []

    relationship_map = await get_relationship_status_map(
        firebase_uid,
        [profile.get("firebase_uid") for profile in profile_docs if profile.get("firebase_uid")],
    )

    public_cards = [
        build_public_profile_card(profile, relationship_map.get(profile.get("firebase_uid"), FRIEND_RELATIONSHIP_NONE))
        for profile in profile_docs
        if profile.get("id")
    ]
    return sort_public_profile_cards(public_cards, search_query)[:10]


@api_router.post("/friends/requests", response_model=FriendActionResponse)
async def send_friend_request(
    input: FriendRequestCreate,
    firebase_uid: str = Header(..., alias="X-Firebase-UID"),
):
    sender_profile = await get_profile_document_by_uid(firebase_uid)
    if not sender_profile:
        raise HTTPException(status_code=404, detail="Önce profilini oluşturmalısın")

    receiver_profile = await db.profiles.find_one({"id": input.to_profile_id}, {"_id": 0})
    if not receiver_profile:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı")

    receiver_uid = receiver_profile.get("firebase_uid")
    if not receiver_uid:
        raise HTTPException(status_code=400, detail="Bu kullanıcıya şu anda istek gönderilemiyor")

    if receiver_uid == firebase_uid:
        raise HTTPException(status_code=400, detail="Kendine arkadaşlık isteği gönderemezsin")

    if await are_users_friends(firebase_uid, receiver_uid):
        raise HTTPException(status_code=409, detail="Bu kullanıcı zaten arkadaşın")

    reverse_pending_request = await db.friend_requests.find_one(
        {
            "from_uid": receiver_uid,
            "to_uid": firebase_uid,
            "status": FRIEND_REQUEST_PENDING,
        },
        {"_id": 0, "id": 1},
    )
    if reverse_pending_request:
        raise HTTPException(status_code=409, detail="Bu kullanıcı sana zaten arkadaşlık isteği gönderdi")

    existing_requests = await db.friend_requests.find(
        {"from_uid": firebase_uid, "to_uid": receiver_uid},
        {"_id": 0},
    ).sort("created_at", -1).to_list(1)
    existing_request = existing_requests[0] if existing_requests else None

    if existing_request and existing_request.get("status") == FRIEND_REQUEST_PENDING:
        raise HTTPException(status_code=409, detail="Bu kullanıcıya zaten bekleyen bir istek gönderdin")

    if existing_request and existing_request.get("status") == FRIEND_REQUEST_ACCEPTED:
        raise HTTPException(status_code=409, detail="Bu kullanıcı zaten arkadaşın")

    now = datetime.now(timezone.utc).isoformat()
    request_snapshot = build_friend_request_snapshot(sender_profile)

    if existing_request and existing_request.get("status") == FRIEND_REQUEST_REJECTED:
        await db.friend_requests.update_one(
            {"id": existing_request["id"]},
            {
                "$set": {
                    "status": FRIEND_REQUEST_PENDING,
                    "created_at": now,
                    **request_snapshot,
                }
            },
        )
    else:
        friend_request_doc = {
            "id": str(uuid.uuid4()),
            "from_uid": firebase_uid,
            "to_uid": receiver_uid,
            "status": FRIEND_REQUEST_PENDING,
            "created_at": now,
            **request_snapshot,
        }
        try:
            await db.friend_requests.insert_one(friend_request_doc)
        except DuplicateKeyError as exc:
            raise HTTPException(status_code=409, detail="Bu kullanıcıya zaten bekleyen bir istek gönderdin") from exc

    return FriendActionResponse(success=True, status=FRIEND_REQUEST_PENDING, detail="Arkadaşlık isteği gönderildi")


@api_router.get("/friends/requests/incoming", response_model=List[FriendRequestNotification])
async def get_incoming_friend_requests(firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    request_docs = await db.friend_requests.find(
        {"to_uid": firebase_uid, "status": FRIEND_REQUEST_PENDING},
        {"_id": 0},
    ).sort("created_at", -1).to_list(200)

    notifications = []
    for request_doc in request_docs:
        created_at = request_doc.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        sender_handle = normalize_existing_handle(request_doc.get("from_handle"))
        sender_username = (request_doc.get("from_username") or "").strip() or "Bilinmeyen Kullanıcı"
        notifications.append(
            FriendRequestNotification(
                id=request_doc["id"],
                status=request_doc.get("status", FRIEND_REQUEST_PENDING),
                created_at=created_at,
                from_display_name=format_handle_display(sender_handle) or sender_username,
                from_username=sender_username,
                from_handle=sender_handle,
                from_handle_display=format_handle_display(sender_handle),
                from_avatar_url=request_doc.get("from_avatar_url") or None,
            )
        )

    return notifications


@api_router.post("/friends/requests/{request_id}/accept", response_model=FriendActionResponse)
async def accept_friend_request(request_id: str, firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    friend_request = await db.friend_requests.find_one({"id": request_id}, {"_id": 0})
    if not friend_request or friend_request.get("to_uid") != firebase_uid:
        raise HTTPException(status_code=404, detail="Arkadaşlık isteği bulunamadı")

    if friend_request.get("status") != FRIEND_REQUEST_PENDING:
        raise HTTPException(status_code=409, detail="Bu istek artık beklemede değil")

    now = datetime.now(timezone.utc).isoformat()
    sender_uid = friend_request.get("from_uid")
    receiver_uid = friend_request.get("to_uid")

    for user_uid, friend_uid in [(sender_uid, receiver_uid), (receiver_uid, sender_uid)]:
        try:
            await db.friends.insert_one(
                {
                    "id": str(uuid.uuid4()),
                    "user_uid": user_uid,
                    "friend_uid": friend_uid,
                    "created_at": now,
                }
            )
        except DuplicateKeyError:
            continue

    await db.friend_requests.update_one(
        {"id": request_id},
        {"$set": {"status": FRIEND_REQUEST_ACCEPTED, "resolved_at": now}},
    )
    return FriendActionResponse(success=True, status=FRIEND_REQUEST_ACCEPTED, detail="Arkadaşlık isteği kabul edildi")


@api_router.post("/friends/requests/{request_id}/reject", response_model=FriendActionResponse)
async def reject_friend_request(request_id: str, firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    friend_request = await db.friend_requests.find_one({"id": request_id}, {"_id": 0})
    if not friend_request or friend_request.get("to_uid") != firebase_uid:
        raise HTTPException(status_code=404, detail="Arkadaşlık isteği bulunamadı")

    if friend_request.get("status") != FRIEND_REQUEST_PENDING:
        raise HTTPException(status_code=409, detail="Bu istek artık beklemede değil")

    await db.friend_requests.update_one(
        {"id": request_id},
        {"$set": {"status": FRIEND_REQUEST_REJECTED, "resolved_at": datetime.now(timezone.utc).isoformat()}},
    )
    return FriendActionResponse(success=True, status=FRIEND_REQUEST_REJECTED, detail="Arkadaşlık isteği reddedildi")


@api_router.get("/friends", response_model=List[PublicProfileCard])
async def get_friends(firebase_uid: str = Header(..., alias="X-Firebase-UID")):
    friend_docs = await db.friends.find(
        {"user_uid": firebase_uid},
        {"_id": 0, "friend_uid": 1},
    ).to_list(500)
    friend_uids = [friend_doc.get("friend_uid") for friend_doc in friend_docs if friend_doc.get("friend_uid")]

    if not friend_uids:
        return []

    profile_docs = await db.profiles.find(
        {"firebase_uid": {"$in": friend_uids}},
        {
            "_id": 0,
            "id": 1,
            "firebase_uid": 1,
            "username": 1,
            "name": 1,
            "handle": 1,
            "avatar_url": 1,
            "study_goal": 1,
            "daily_study_hours": 1,
        },
    ).to_list(500)

    friend_cards = [build_public_profile_card(profile, FRIEND_RELATIONSHIP_FRIENDS) for profile in profile_docs if profile.get("id")]
    return sorted(friend_cards, key=lambda card: ((card.display_name or "").lower(), (card.username or "").lower()))

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

# Backward-compatible alias for environments using old create path
@api_router.post("/program/create", response_model=Program)
async def create_program_alias(input: ProgramCreate):
    return await create_program(input)

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

    room_type = normalize_room_type(input.room_type)
    room_password = normalize_room_password(input.room_password)
    if room_type == ROOM_TYPE_PRIVATE and (not room_password or len(room_password) < ROOM_PASSWORD_MIN_LENGTH):
        raise HTTPException(status_code=400, detail=ROOM_PASSWORD_VALIDATION_MESSAGE)
    
    room_obj = Room(
        name=input.name,
        owner_id=input.owner_id or str(uuid.uuid4()),
        room_type=room_type,
        is_private=room_type == ROOM_TYPE_PRIVATE,
    )
    
    # Add owner as first participant
    owner = RoomParticipant(
        id=room_obj.owner_id,
        name=input.owner_name,
        avatar_url=input.owner_avatar_url,
        study_field=input.owner_study_field
    )
    room_obj.participants.append(owner)
    
    doc = room_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['room_password_hash'] = hash_room_password(room_password) if room_type == ROOM_TYPE_PRIVATE else None
    
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

    room_type = normalize_room_type(room.get("room_type"))
    is_private = room_type == ROOM_TYPE_PRIVATE or bool(room.get("room_password_hash"))
    if is_private and not verify_room_password(input.room_password, room.get("room_password_hash")):
        raise HTTPException(
            status_code=403,
            detail=ROOM_PASSWORD_REQUIRED_MESSAGE if not normalize_room_password(input.room_password) else ROOM_PASSWORD_INVALID_MESSAGE,
        )

    participant_id = input.user_id or str(uuid.uuid4())
    participants = room.get("participants", [])
    existing_index = next((index for index, participant in enumerate(participants) if participant.get("id") == participant_id), None)

    participant_doc = RoomParticipant(
        id=participant_id,
        name=input.user_name,
        avatar_url=input.user_avatar_url,
        study_field=input.user_study_field
    ).model_dump()

    participant_added = existing_index is None
    if participant_added:
        participants.append(participant_doc)
    else:
        participants[existing_index] = {**participants[existing_index], **participant_doc}

    await db.rooms.update_one(
        {"code": input.room_code.upper()},
        {"$set": {"participants": participants}}
    )

    if participant_added:
        await create_system_room_message(room["id"], f"{input.user_name} odaya katıldı")
    
    updated_room = await db.rooms.find_one({"code": input.room_code.upper()}, {"_id": 0})
    return format_room_document(updated_room)

@api_router.post("/rooms/{room_id}/leave")
async def leave_room(room_id: str, input: RoomLeave):
    room = await db.rooms.find_one({"id": room_id}, {"_id": 0})
    if not room:
        return {"success": False, "error": "Oda bulunamadı"}

    participants = room.get("participants", [])
    updated_participants = [participant for participant in participants if participant.get("id") != input.user_id]

    if len(updated_participants) == len(participants):
        return {"success": False, "removed": False}

    await db.rooms.update_one(
        {"id": room_id},
        {"$set": {"participants": updated_participants}}
    )

    if input.user_name:
        await create_system_room_message(room_id, f"{input.user_name} odadan ayrıldı")

    return {"success": True, "removed": True}

@api_router.get("/rooms/{room_id}", response_model=Room)
async def get_room(room_id: str):
    room = await db.rooms.find_one({"id": room_id}, {"_id": 0})
    return format_room_document(room)

@api_router.get("/rooms/code/{room_code}", response_model=Room)
async def get_room_by_code(room_code: str):
    room = await db.rooms.find_one({"code": room_code.upper()}, {"_id": 0})
    if not room:
        raise HTTPException(status_code=404, detail="Oda bulunamadı")
    return format_room_document(room)

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


# ============ STUDY SESSION ENDPOINTS ============

@api_router.post("/study-sessions/start")
async def start_study_session(session_data: StudySessionStart):
    """Start a new study session or return existing active session"""
    try:
        # Check if there's already an active session for this user in this room
        existing_session = await db.study_sessions.find_one({
            "firebase_uid": session_data.firebase_uid,
            "room_id": session_data.room_id,
            "is_completed": False
        })
        
        if existing_session:
            # Remove MongoDB _id before returning
            if '_id' in existing_session:
                del existing_session['_id']
            return existing_session
        
        # Create new session
        session_dict = session_data.model_dump()
        session_obj = StudySession(**session_dict)
        session_dump = session_obj.model_dump()
        await db.study_sessions.insert_one(session_dump.copy())
        
        # Return without MongoDB _id
        return session_dump
    except Exception as e:
        logging.error(f"Error starting study session: {e}")
        raise


@api_router.put("/study-sessions/{session_id}/update")
async def update_study_session(session_id: str, update_data: StudySessionUpdate):
    """Auto-save study session progress"""
    try:
        await db.study_sessions.update_one(
            {"id": session_id},
            {
                "$set": {
                    "accumulated_seconds": update_data.accumulated_seconds,
                    "last_saved_at": datetime.now(timezone.utc)
                }
            }
        )
        
        updated_session = await db.study_sessions.find_one({"id": session_id})
        if not updated_session:
            return {"error": "Session not found"}
        
        # Remove MongoDB _id
        if '_id' in updated_session:
            del updated_session['_id']
        
        return updated_session
    except Exception as e:
        logging.error(f"Error updating study session: {e}")
        raise


@api_router.put("/study-sessions/{session_id}/complete")
async def complete_study_session(session_id: str, complete_data: StudySessionComplete):
    """Mark study session as completed"""
    try:
        await db.study_sessions.update_one(
            {"id": session_id},
            {
                "$set": {
                    "accumulated_seconds": complete_data.accumulated_seconds,
                    "is_completed": True,
                    "last_saved_at": datetime.now(timezone.utc)
                }
            }
        )
        
        completed_session = await db.study_sessions.find_one({"id": session_id})
        if not completed_session:
            return {"error": "Session not found"}

        if int(completed_session.get("accumulated_seconds") or 0) > 0:
            await update_profile_streak(completed_session.get("firebase_uid"))
        
        # Remove MongoDB _id
        if '_id' in completed_session:
            del completed_session['_id']
        
        return completed_session
    except Exception as e:
        logging.error(f"Error completing study session: {e}")
        raise


@api_router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard():
    """Return global leaderboard ranked by total accumulated study time"""
    try:
        aggregation_pipeline = [
            {
                "$group": {
                    "_id": "$firebase_uid",
                    "total_seconds": {"$sum": "$accumulated_seconds"}
                }
            },
            {"$sort": {"total_seconds": -1}}
        ]

        grouped_results = await db.study_sessions.aggregate(aggregation_pipeline).to_list(1000)

        if not grouped_results:
            return []

        user_ids = [entry.get("_id") for entry in grouped_results if entry.get("_id")]

        profile_docs = await db.profiles.find(
            {"firebase_uid": {"$in": user_ids}},
            {"_id": 0, "firebase_uid": 1, "username": 1, "name": 1, "handle": 1, "avatar_url": 1}
        ).to_list(1000)
        profile_identity_map = {
            profile["firebase_uid"]: {
                "user_name": resolve_public_profile_name(profile),
                "avatar_url": profile.get("avatar_url")
            }
            for profile in profile_docs
            if profile.get("firebase_uid")
        }

        unresolved_user_ids = [uid for uid in user_ids if uid not in profile_identity_map]
        room_identity_map = {}

        if unresolved_user_ids:
            room_participant_pipeline = [
                {"$unwind": "$participants"},
                {"$match": {"participants.id": {"$in": unresolved_user_ids}}},
                {
                    "$project": {
                        "_id": 0,
                        "participant_id": "$participants.id",
                        "participant_name": "$participants.name",
                        "participant_avatar_url": "$participants.avatar_url",
                        "created_at": 1
                    }
                },
                {"$sort": {"created_at": -1}}
            ]

            room_participants = await db.rooms.aggregate(room_participant_pipeline).to_list(2000)
            for participant in room_participants:
                participant_id = participant.get("participant_id")
                participant_name = participant.get("participant_name")
                if participant_id and participant_id not in room_identity_map:
                    room_identity_map[participant_id] = {
                        "user_name": participant_name,
                        "avatar_url": participant.get("participant_avatar_url")
                    }

        leaderboard_entries = []
        for index, entry in enumerate(grouped_results, start=1):
            user_id = entry.get("_id", "")
            total_seconds = int(entry.get("total_seconds", 0))

            fallback_name = "Bilinmeyen Kullanıcı"
            profile_identity = profile_identity_map.get(user_id, {})
            room_identity = room_identity_map.get(user_id, {})
            user_name = profile_identity.get("user_name") or room_identity.get("user_name") or fallback_name
            avatar_url = (
                profile_identity.get("avatar_url")
                if user_id in profile_identity_map
                else room_identity.get("avatar_url")
            )

            leaderboard_entries.append(
                LeaderboardEntry(
                    rank=index,
                    user_id=user_id,
                    user_name=user_name,
                    avatar_url=avatar_url,
                    total_seconds=total_seconds
                )
            )

        return leaderboard_entries
    except Exception as e:
        logging.error(f"Error loading leaderboard: {e}")
        raise


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

@app.on_event("startup")
async def ensure_profile_indexes():
    try:
        await db.profiles.create_index(
            "handle",
            unique=True,
            partialFilterExpression={"handle": {"$type": "string"}},
        )
        await db.friend_requests.create_index(
            [("from_uid", 1), ("to_uid", 1), ("status", 1)],
            unique=True,
            partialFilterExpression={"status": FRIEND_REQUEST_PENDING},
        )
        await db.friend_requests.create_index([("to_uid", 1), ("status", 1)])
        await db.friends.create_index([("user_uid", 1), ("friend_uid", 1)], unique=True)
    except Exception as exc:
        logger.warning("Profile/friend indexes could not be created: %s", exc)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()