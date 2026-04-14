from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
import uuid
import time
import logging
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from process_video import process_video
import stripe
import asyncio
import yt_dlp
import json
import traceback
from auth_db import (
    init_db,
    create_user,
    authenticate_user,
    create_session,
    get_user_by_token,
    deduct_tokens,
    add_tokens,
    refund_tokens,
    get_user_by_login,
    create_password_reset_token,
    consume_password_reset_token,
    save_youtube_token,
    get_youtube_token,
    ensure_token_allowance,
    get_plan_catalog,
    set_subscription_plan,
    upsert_video_job,
    get_video_job,
)
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# YouTube OAuth settings
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")
CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), "client_secrets.json")
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_REDIRECT_URI = f"{FRONTEND_BASE_URL}/?tab=account&mode=youtube_callback" # Default if not specified

# In production this must be your deployed backend URL callback endpoint.
YOUTUBE_BACKEND_REDIRECT_URI = os.getenv(
    "YOUTUBE_BACKEND_REDIRECT_URI",
    "http://localhost:8000/auth/youtube/callback",
)

jobs = {}
OUTPUT_FOLDER = "output"
UPLOAD_FOLDER = "uploads"


def _job_dir(job_id: str) -> str:
    return os.path.join(OUTPUT_FOLDER, job_id)


def _job_status_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "status.json")


def _persist_job_state(job_id: str, payload: dict) -> None:
    safe_payload = dict(payload)
    safe_payload.pop("session_token", None)
    safe_payload["job_id"] = job_id
    upsert_video_job(job_id, safe_payload, safe_payload.get("owner_user_id"))


def _load_persisted_job_state(job_id: str) -> dict | None:
    try:
        return get_video_job(job_id)
    except Exception:
        logger.exception("Failed to load persisted job state for %s", job_id)
    return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autoeditor.api")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PROCESS_TOKEN_COST = int(os.getenv("PROCESS_TOKEN_COST", "25"))
STARTING_TOKENS = int(os.getenv("STARTING_TOKENS", "120"))
RESET_TOKEN_DEBUG = os.getenv("RESET_TOKEN_DEBUG", "false").lower() == "true"
DAILY_FREE_TOKENS = int(os.getenv("FREE_DAILY_TOKENS", "100"))

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
if STRIPE_SECRET_KEY and STRIPE_SECRET_KEY.lower() not in ["none", "fixme", "sk_test_..."]:
    stripe.api_key = STRIPE_SECRET_KEY
else:
    STRIPE_SECRET_KEY = None

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    email: str
    password: str = Field(min_length=8, max_length=100)


class LoginRequest(BaseModel):
    login: str
    password: str


class ForgotPasswordRequest(BaseModel):
    login: str


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(min_length=8, max_length=100)


class TopupRequest(BaseModel):
    tokens: int = Field(gt=0, le=10000)


class CheckoutSessionRequest(BaseModel):
    tokens: int = Field(gt=0, le=10000)


class CheckoutConfirmRequest(BaseModel):
    session_id: str


class PlanChangeRequest(BaseModel):
    plan: str

class ProcessUrlRequest(BaseModel):
    url: str
    use_ai: bool = True
    export_format: str = "16:9"
    video_filter: str = "none"

class ExportYoutubeRequest(BaseModel):
    job_id: str
    title: str
    description: str
    export_format: str = "16:9"


TOKEN_PACKS = [
    {"tokens": 2000, "amount_usd": 5.0},
    {"tokens": 5000, "amount_usd": 10.0},
    {"tokens": 15000, "amount_usd": 25.0},
]


def _token_pack_for(tokens: int) -> dict | None:
    for pack in TOKEN_PACKS:
        if pack["tokens"] == tokens:
            return pack
    return None


def _send_password_reset_email(to_email: str, username: str, token: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM):
        logger.warning("SMTP is not configured; cannot send reset email for %s", to_email)
        return False

    reset_url = f"{FRONTEND_BASE_URL}/?tab=account&mode=reset&token={token}"
    msg = EmailMessage()
    msg["Subject"] = "AutoEditor Pro password reset"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(
        (
            f"Hi {username},\n\n"
            "We received a request to reset your AutoEditor Pro password.\n"
            f"Reset link: {reset_url}\n\n"
            "If you did not request this, you can ignore this message.\n"
            "This link expires in 30 minutes.\n"
        )
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        return True
    except Exception as e:
        logger.error("SMTP failed to send email to %s: %s", to_email, e)
        return False


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    return parts[1].strip()


def _require_user(authorization: str | None) -> dict:
    token = _extract_bearer_token(authorization)
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

def bg_process_url(job_id: str, url: str, output_folder: str, output_dir: str, user_id: int, token_cost: int, use_ai: bool = True, export_format: str = "16:9", video_filter: str = "none"):
    started_at = time.perf_counter()
    owner_user_id = jobs.get(job_id, {}).get("owner_user_id")
    session_token = jobs.get(job_id, {}).get("session_token")
    video_path = ""
    
    try:
        # Try multiple browsers one by one for cookies to avoid "keyring" errors
        browsers_to_try = [('chrome',), ('edge',), ('brave',), ('vivaldi',), None]
        download_success = False
        
        for browser_opt in browsers_to_try:
            try:
                current_opts = {
                    'format': 'best',
                    'outtmpl': os.path.join(output_folder, f'{job_id}.%(ext)s'),
                    'noplaylist': True,
                    'quiet': True,
                }
                if browser_opt:
                    current_opts['cookiesfrombrowser'] = browser_opt
                
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
                    download_success = True
                    break
            except Exception as e:
                logger.warning(f"Failed to download using browser {browser_opt}: {e}")
                continue
                
        if not download_success:
            raise Exception("Could not download video. YouTube is blocking the request. Please ensure you are logged in to YouTube in Chrome or Edge and close the browser before trying again.")
            
        jobs[job_id]["stage"] = "processing downloaded video"
        _persist_job_state(job_id, jobs[job_id])
        process_result = process_video(video_path, output_dir, use_ai=use_ai, export_format=export_format, video_filter=video_filter)
        output_video_path = process_result.get("output_video_path") if process_result else None
        summary_text = process_result.get("summary_text") if process_result else None
        summary_data = process_result.get("summary_data") if process_result else None
        
        if not output_video_path or not os.path.exists(output_video_path):
            updated_tokens = refund_tokens(user_id, token_cost)
            jobs[job_id].update({
                "status": "error",
                "stage": "completed",
                "message": "No cheers detected. Video processed but no highlights generated.",
                "summary": summary_text,
                "summary_data": summary_data,
                "download_url": None,
                "processing_seconds": round(time.perf_counter() - started_at, 2),
                "tokens_refunded": token_cost,
                "tokens_balance": updated_tokens,
            })
            _persist_job_state(job_id, jobs[job_id])
        else:
            current_user = get_user_by_token(session_token) if session_token else None
            jobs[job_id].update({
                "status": "success",
                "stage": "completed",
                "message": "Video processed successfully.",
                "summary": summary_text,
                "summary_data": summary_data,
                "download_url": f"/download/{job_id}/highlight_reel.mp4",
                "processing_seconds": round(time.perf_counter() - started_at, 2),
                "tokens_charged": token_cost,
                "tokens_balance": current_user["tokens"] if current_user else None,
            })
            _persist_job_state(job_id, jobs[job_id])
            
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        updated_tokens = refund_tokens(user_id, token_cost)
        jobs[job_id].update({
            "status": "error",
            "stage": "failed",
            "message": str(e),
            "error_detail": str(e),
            "error_trace": traceback.format_exc(),
            "summary": None,
            "summary_data": None,
            "download_url": None,
            "processing_seconds": round(time.perf_counter() - started_at, 2),
            "tokens_refunded": token_cost,
            "tokens_balance": updated_tokens,
        })
        _persist_job_state(job_id, jobs[job_id])
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

def bg_process_video(job_id: str, video_path: str, output_dir: str, user_id: int, token_cost: int, use_ai: bool = True, export_format: str = "16:9"):
    started_at = time.perf_counter()
    owner_user_id = jobs.get(job_id, {}).get("owner_user_id")
    session_token = jobs.get(job_id, {}).get("session_token")
    try:
        process_result = process_video(video_path, output_dir, use_ai=use_ai, export_format=export_format)
        output_video_path = process_result.get("output_video_path") if process_result else None
        summary_text = process_result.get("summary_text") if process_result else None
        summary_data = process_result.get("summary_data") if process_result else None
        
        if not output_video_path or not os.path.exists(output_video_path):
            updated_tokens = refund_tokens(user_id, token_cost)
            jobs[job_id] = {
                "status": "error",
                "stage": "completed",
                "message": "No cheers detected. Video processed but no highlights generated.",
                "summary": summary_text,
                "summary_data": summary_data,
                "download_url": None,
                "processing_seconds": round(time.perf_counter() - started_at, 2),
                "tokens_refunded": token_cost,
                "tokens_balance": updated_tokens,
                "owner_user_id": owner_user_id,
                "session_token": session_token,
            }
            _persist_job_state(job_id, jobs[job_id])
        else:
            current_user = get_user_by_token(session_token) if session_token else None
            jobs[job_id] = {
                "status": "success",
                "stage": "completed",
                "message": "Video processed successfully.",
                "summary": summary_text,
                "summary_data": summary_data,
                "download_url": f"/download/{job_id}/highlight_reel.mp4",
                "processing_seconds": round(time.perf_counter() - started_at, 2),
                "tokens_charged": token_cost,
                "tokens_balance": current_user["tokens"] if current_user else None,
                "owner_user_id": owner_user_id,
                "session_token": session_token,
            }
            _persist_job_state(job_id, jobs[job_id])
            
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        updated_tokens = refund_tokens(user_id, token_cost)
        jobs[job_id] = {
            "status": "error",
            "stage": "failed",
            "message": str(e),
            "error_detail": str(e),
            "error_trace": traceback.format_exc(),
            "summary": None,
            "summary_data": None,
            "download_url": None,
            "processing_seconds": round(time.perf_counter() - started_at, 2),
            "tokens_refunded": token_cost,
            "tokens_balance": updated_tokens,
            "owner_user_id": owner_user_id,
            "session_token": session_token,
        }
        _persist_job_state(job_id, jobs[job_id])
    finally:
        # Clean up the original upload
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

app = FastAPI(title="Automated Video Editor API", description="API to process videos and extract cheering moments.")

# CORS settings to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
init_db()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Automated Video Editor API!",
        "max_upload_mb": MAX_UPLOAD_MB,
        "process_token_cost": PROCESS_TOKEN_COST,
    }


class TestEmailRequest(BaseModel):
    email: str

@app.post("/auth/test-email")
async def test_email_endpoint(payload: TestEmailRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    # Check if SMTP is configured
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise HTTPException(status_code=400, detail="SMTP is not configured. Check your .env file.")
        
    msg = EmailMessage()
    msg["Subject"] = "AutoEditor Pro - Test Email"
    msg["From"] = SMTP_FROM
    msg["To"] = payload.email
    msg.set_content(f"Hello {user['username']},\n\nYour SMTP configuration is working correctly! 🔥\n\n- AutoEditor Pro Team")
    
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        return {"message": f"Test email sent successfully to {payload.email}"}
    except Exception as e:
        logger.error(f"Test email failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


@app.post("/auth/register")
async def register(payload: RegisterRequest):
    try:
        user = create_user(payload.username, payload.email, payload.password, starting_tokens=STARTING_TOKENS)
    except Exception:
        raise HTTPException(status_code=409, detail="Email or username already registered")

    ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
    token = create_session(user["id"])
    enriched_user = get_user_by_token(token)
    return {
        "access_token": token,
        "user": enriched_user,
    }


@app.post("/auth/login")
async def login(payload: LoginRequest):
    user = authenticate_user(payload.login, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
    token = create_session(user["id"])
    enriched_user = get_user_by_token(token)
    return {
        "access_token": token,
        "user": enriched_user,
    }


@app.post("/auth/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    user = get_user_by_login(payload.login)
    # Keep response shape fixed to avoid account enumeration.
    if not user:
        return {"message": "If this account exists, a reset message has been sent."}

    token = create_password_reset_token(user["id"], expires_minutes=30)
    sent = False
    try:
        sent = _send_password_reset_email(user["email"], user["username"], token)
    except Exception:
        logger.exception("Could not send reset email to %s", user["email"])

    response = {"message": "If this account exists, a reset message has been sent."}
    if RESET_TOKEN_DEBUG:
        response["debug_reset_token"] = token
        response["email_sent"] = sent
    return response


@app.post("/auth/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    ok = consume_password_reset_token(payload.token, payload.password)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    return {"message": "Password has been reset successfully."}


@app.get("/auth/me")
async def me(authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    # Refresh trial/free/paid allowances automatically on profile fetch.
    try:
        ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
    except Exception as e:
        logger.error(f"Daily token refresh failed: {e}")
        
    # Fetch updated user info
    user = get_user_by_token(_extract_bearer_token(authorization))
    return user

@app.get("/auth/youtube/login")
async def youtube_login(authorization: str | None = Header(default=None)):
    user = _require_user(authorization)

    if user.get("effective_plan") not in {"trial", "pro"}:
        raise HTTPException(
            status_code=403,
            detail="YouTube publishing is available on Pro plan only (or during full trial).",
        )
    
    if not os.path.exists(CLIENT_SECRETS_FILE):
        raise HTTPException(status_code=503, detail="YouTube API is not configured on the server (missing client_secrets.json)")

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=YOUTUBE_UPLOAD_SCOPE,
        redirect_uri=YOUTUBE_BACKEND_REDIRECT_URI
    )
    
    # Fixed code verifier (at least 43 chars) to avoid PKCE 'Missing code verifier' error
    # We use a static one for local development simplicity
    static_verifier = "static_verifier_for_local_development_purposes_only_1234567890"
    flow.code_verifier = static_verifier
    
    # Generate auth URL. 'state' can be used to pass the session token back securely.
    session_token = _extract_bearer_token(authorization)
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=session_token,
        prompt='consent'
    )
    
    return {"auth_url": auth_url}


@app.post("/auth/youtube/disconnect")
async def youtube_disconnect(authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    save_youtube_token(user["id"], None)
    return {"message": "YouTube account disconnected successfully"}

@app.get("/auth/youtube/callback")
async def youtube_callback(code: str, state: str):
    # 'state' contains our session_token
    user = get_user_by_token(state)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session state")

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=YOUTUBE_UPLOAD_SCOPE,
        redirect_uri=YOUTUBE_BACKEND_REDIRECT_URI
    )
    
    # Use the same 50-char verifier in the callback
    static_verifier = "static_verifier_for_local_development_purposes_only_1234567890"
    
    flow.fetch_token(code=code, code_verifier=static_verifier)
    credentials = flow.credentials
    
    # Save the credentials JSON to the DB
    # We explicitly include client_id and client_secret to ensure refresh works
    creds_data = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id or flow.client_config.get('client_id'),
        'client_secret': credentials.client_secret or flow.client_config.get('client_secret'),
        'scopes': credentials.scopes
    }
    save_youtube_token(user["id"], json.dumps(creds_data))
    
    # Validation: If we still don't have a refresh token, inform the user during the redirect
    if not credentials.refresh_token:
        # We still redirect to a success-ish page but with a warning or just handle it in redirect.html
        # Better: let's use a specialized redirect for failures
        pass

    # Redirect back to frontend account tab
    return FileResponse(os.path.join(os.path.dirname(__file__), "redirect.html"))


@app.get("/billing/pricing")
async def pricing():
    catalog = get_plan_catalog()
    return {
        "process_token_cost": PROCESS_TOKEN_COST,
        "token_packs": TOKEN_PACKS,
        "plans": catalog,
        "gateway": "stripe" if STRIPE_SECRET_KEY else "manual",
    }


@app.get("/billing/plans")
async def billing_plans(authorization: str | None = Header(default=None)):
    catalog = get_plan_catalog()
    current_user = None
    if authorization:
        try:
            user = _require_user(authorization)
            ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
            current_user = get_user_by_token(_extract_bearer_token(authorization))
        except HTTPException:
            current_user = None
    return {
        "catalog": catalog,
        "current_user": current_user,
    }


@app.post("/billing/plan")
async def change_plan(payload: PlanChangeRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    plan = payload.plan.strip().lower()

    if plan not in {"free", "plus", "pro"}:
        raise HTTPException(status_code=400, detail="Invalid plan. Supported plans: free, plus, pro")

    # This endpoint updates local entitlement state.
    # In production, call it only after successful subscription checkout/webhook.
    active = plan in {"plus", "pro"}
    try:
        updated_user = set_subscription_plan(user["id"], plan, active=active)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "message": "Plan updated successfully",
        "user": updated_user,
    }


@app.post("/billing/topup")
async def topup(payload: TopupRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    amount = round(payload.tokens * 0.05, 2)
    balance = add_tokens(user["id"], payload.tokens, amount_usd=amount, status="paid")
    return {
        "message": "Top-up successful",
        "tokens_added": payload.tokens,
        "amount_usd": amount,
        "tokens_balance": balance,
    }


@app.post("/billing/checkout-session")
async def create_checkout_session(payload: CheckoutSessionRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured yet")

    pack = _token_pack_for(payload.tokens)
    if not pack:
        raise HTTPException(status_code=400, detail="Invalid token pack")

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            customer_email=user["email"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": f"AutoEditor Pro {pack['tokens']} token pack",
                        },
                        "unit_amount": int(pack["amount_usd"] * 100),
                    },
                    "quantity": 1,
                }
            ],
            success_url=f"{FRONTEND_BASE_URL}/?tab=pricing&checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_BASE_URL}/?tab=pricing&checkout=cancel",
            metadata={
                "user_id": str(user["id"]),
                "tokens": str(pack["tokens"]),
                "amount_usd": str(pack["amount_usd"]),
            },
        )
    except Exception:
        logger.exception("Stripe session creation failed for user %s", user["id"])
        raise HTTPException(status_code=502, detail="Unable to create checkout session")

    return {"checkout_url": session.url}


@app.post("/billing/checkout-confirm")
async def confirm_checkout(payload: CheckoutConfirmRequest, authorization: str | None = Header(default=None)):
    user = _require_user(authorization)
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe is not configured yet")

    try:
        session = stripe.checkout.Session.retrieve(payload.session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid checkout session")

    if session.get("payment_status") != "paid":
        raise HTTPException(status_code=400, detail="Payment not completed")

    metadata = session.get("metadata") or {}
    session_user_id = int(metadata.get("user_id", "0"))
    tokens = int(metadata.get("tokens", "0"))
    amount_usd = float(metadata.get("amount_usd", "0"))

    if session_user_id != user["id"]:
        raise HTTPException(status_code=403, detail="Checkout does not belong to this user")
    if tokens <= 0:
        raise HTTPException(status_code=400, detail="Invalid token amount in checkout session")

    balance = add_tokens(
        user["id"],
        tokens,
        amount_usd=amount_usd,
        status="paid",
        gateway_session_id=payload.session_id,
    )
    return {
        "message": "Top-up successful",
        "tokens_added": tokens,
        "amount_usd": amount_usd,
        "tokens_balance": balance,
    }

@app.post("/process")
async def process_video_endpoint(
    file: UploadFile = File(...),
    use_ai: bool = True,
    export_format: str = "16:9",
    background_tasks: BackgroundTasks = None,
    authorization: str | None = Header(default=None),
):
    user = _require_user(authorization)
    ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
    user = get_user_by_token(_extract_bearer_token(authorization))

    if not file.content_type or not file.content_type.startswith("video/"):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    if user["tokens"] < PROCESS_TOKEN_COST:
        raise HTTPException(
            status_code=402,
            detail=f"Not enough tokens. You need at least {PROCESS_TOKEN_COST} tokens to process a video.",
        )

    if not deduct_tokens(user["id"], PROCESS_TOKEN_COST):
        raise HTTPException(status_code=402, detail="Not enough tokens. Please top up and try again.")
    
    # Create unique filename
    unique_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_filename = f"{unique_id}{file_extension}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    # Save upload without size restrictions
    written_bytes = 0
    chunk_size = 1024 * 1024
    with open(video_path, "wb") as buffer:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            written_bytes += len(chunk)
            buffer.write(chunk)
    await file.close()
    
    # Process the video in background
    output_dir = os.path.join(OUTPUT_FOLDER, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    jobs[unique_id] = {
        "status": "processing",
        "stage": "queued",
        "message": "Analyzing audio & video...",
        "download_url": None,
        "owner_user_id": user["id"],
        "session_token": _extract_bearer_token(authorization),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_size_mb": round(written_bytes / (1024 * 1024), 2),
        "tokens_charged": PROCESS_TOKEN_COST,
        "tokens_balance": max(user["tokens"] - PROCESS_TOKEN_COST, 0),
    }
    _persist_job_state(unique_id, jobs[unique_id])
    logger.info(
        "Queued job %s by user %s (%s, %.2f MB)",
        unique_id,
        user["id"],
        file.filename,
        written_bytes / (1024 * 1024),
    )
    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Background task manager is unavailable")
    background_tasks.add_task(bg_process_video, unique_id, video_path, output_dir, user["id"], PROCESS_TOKEN_COST, use_ai=use_ai, export_format=export_format)
    return {
        "job_id": unique_id,
        "status": "processing",
        "stage": "queued",
        "message": "Processing started.",
        "max_upload_mb": MAX_UPLOAD_MB,
        "tokens_charged": PROCESS_TOKEN_COST,
        "tokens_balance": max(user["tokens"] - PROCESS_TOKEN_COST, 0),
    }

@app.post("/process-url")
async def process_url_endpoint(
    payload: ProcessUrlRequest,
    background_tasks: BackgroundTasks = None,
    authorization: str | None = Header(default=None),
):
    user = _require_user(authorization)
    ensure_token_allowance(user["id"], DAILY_FREE_TOKENS)
    user = get_user_by_token(_extract_bearer_token(authorization))

    if user["tokens"] < PROCESS_TOKEN_COST:
        raise HTTPException(status_code=402, detail=f"Not enough tokens. You need at least {PROCESS_TOKEN_COST} to process a video.")

    if not deduct_tokens(user["id"], PROCESS_TOKEN_COST):
        raise HTTPException(status_code=402, detail="Not enough tokens. Please top up and try again.")
    
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(OUTPUT_FOLDER, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    jobs[unique_id] = {
        "status": "processing",
        "stage": "queued",
        "message": "Downloading video from URL...",
        "download_url": None,
        "owner_user_id": user["id"],
        "session_token": _extract_bearer_token(authorization),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_size_mb": None,
        "tokens_charged": PROCESS_TOKEN_COST,
        "tokens_balance": max(user["tokens"] - PROCESS_TOKEN_COST, 0),
    }
    _persist_job_state(unique_id, jobs[unique_id])
    
    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Background task manager is unavailable")
    
    background_tasks.add_task(
        bg_process_url, 
        unique_id, 
        payload.url, 
        UPLOAD_FOLDER, 
        output_dir, 
        user["id"], 
        PROCESS_TOKEN_COST, 
        use_ai=payload.use_ai, 
        export_format=payload.export_format,
        video_filter=payload.video_filter
    )
    
    return {
        "job_id": unique_id,
        "status": "processing",
        "stage": "queued",
        "message": "Download started.",
        "tokens_charged": PROCESS_TOKEN_COST,
        "tokens_balance": max(user["tokens"] - PROCESS_TOKEN_COST, 0),
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str, authorization: str | None = Header(default=None)):
    try:
        user = _require_user(authorization)
    except HTTPException as e:
        logger.warning(f"[Status] Auth failed for job {job_id}: {e.detail}")
        raise
    
    logger.info(f"[Status] Checking job {job_id} for user {user.get('id')}")
    
    job_state = jobs.get(job_id)
    if not job_state:
        logger.info(f"[Status] Job {job_id} not in memory, checking disk...")
        job_state = _load_persisted_job_state(job_id)
        if job_state:
            logger.info(f"[Status] Job {job_id} loaded from disk, status={job_state.get('status')}")
    else:
        logger.info(f"[Status] Job {job_id} found in memory, status={job_state.get('status')}")
    
    if not job_state:
        logger.warning(f"[Status] Job {job_id} not found (not in memory or disk)")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found (not processing)")

    if job_id not in jobs and job_state.get("status") == "processing":
        logger.warning(f"[Status] Job {job_id} recovered from DB but worker is gone; marking interrupted")
        interrupted = dict(job_state)
        interrupted.update({
            "status": "error",
            "stage": "failed",
            "message": "Processing was interrupted because the backend restarted. Please resubmit the video.",
            "error_detail": "Processing was interrupted because the backend restarted. Please resubmit the video.",
        })
        _persist_job_state(job_id, interrupted)
        job_state = interrupted
    
    owner_id = job_state.get("owner_user_id")
    if owner_id not in (None, user["id"]):
        logger.warning(f"[Status] Job {job_id} ownership mismatch: owner={owner_id}, user={user['id']}")
        raise HTTPException(status_code=403, detail="You do not have access to this job")

    safe = dict(job_state)
    safe.pop("owner_user_id", None)
    safe.pop("session_token", None)
    logger.info(f"[Status] Returning status for job {job_id}: {safe.get('status')}")
    return safe

@app.get("/download/{job_id}/{filename}")
async def download_video(job_id: str, filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, job_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/export/youtube")
async def export_youtube_endpoint(
    payload: ExportYoutubeRequest,
    authorization: str | None = Header(default=None)
):
    user = _require_user(authorization)

    # YouTube publishing is a highest-tier feature after trial.
    if user.get("effective_plan") not in {"trial", "pro"}:
        raise HTTPException(
            status_code=403,
            detail="YouTube publishing is available on Pro plan only (or during full trial).",
        )
    
    job_id = payload.job_id
    
    # Robust Job Check: If not in memory (restored server), check filesystem
    file_path = os.path.join(OUTPUT_FOLDER, job_id, "highlight_reel.mp4")
    
    if job_id not in jobs:
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail="Job results not found on server. Please process again.")
    else:
        if jobs[job_id].get("owner_user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        
    file_path = os.path.join(OUTPUT_FOLDER, job_id, "highlight_reel.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Highlight reel not found for this job")
        
    # Get YouTube credentials from DB
    creds_json = get_youtube_token(user["id"])
    if not creds_json:
        raise HTTPException(status_code=400, detail="YouTube account not connected. Please connect in settings.")
        
    try:
        creds_data = json.loads(creds_json)
        
        # Verify we have the required fields for refresh
        required_fields = ['refresh_token', 'client_id', 'client_secret']
        missing = [f for f in required_fields if not creds_data.get(f)]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"YouTube session is incomplete (missing {', '.join(missing)}). Please disconnect and 'Connect YouTube' again to fix this."
            )

        credentials = Credentials.from_authorized_user_info(creds_data, YOUTUBE_UPLOAD_SCOPE)
        
        youtube = build('youtube', 'v3', credentials=credentials)
        
        # Determine if it is a Short based on format
        is_shorts_format = payload.export_format == "9:16"
        
        # Add #Shorts only for vertical videos
        final_description = payload.description
        if is_shorts_format and "#shorts" not in final_description.lower():
            final_description += "\n\n#Shorts #AIHighlights"

        request_body = {
            'snippet': {
                'title': payload.title,
                'description': final_description,
                'tags': ['shorts', 'sports', 'highlights', 'autoeditor'] if is_shorts_format else ['sports', 'highlights', 'autoeditor'],
                'categoryId': '17' # Sports
            },
            'status': {
                'privacyStatus': 'public', 
                'selfDeclaredMadeForKids': False
            }
        }
        
        media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype='video/mp4')
        
        response = youtube.videos().insert(
            part='snippet,status',
            body=request_body,
            media_body=media
        ).execute()
        
        video_id = response.get('id')
        return {
            "message": "Successfully published to YouTube!",
            "youtube_url": f"https://youtube.com/watch?v={video_id}",
            "is_real": True
        }
        
    except Exception as e:
        logger.exception("YouTube upload failed for user %s", user["id"])
        raise HTTPException(status_code=500, detail=f"YouTube API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
