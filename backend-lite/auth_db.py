import os
import json
import sqlite3
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USING_POSTGRES = DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")

if USING_POSTGRES:
    import psycopg2
    import psycopg2.extras

DB_PATH = os.getenv("AUTH_DB_PATH", os.path.join(os.path.dirname(__file__), "app.db"))

TRIAL_DAYS = int(os.getenv("TRIAL_DAYS", "60"))
TRIAL_DAILY_TOKENS = int(os.getenv("TRIAL_DAILY_TOKENS", "400"))
FREE_DAILY_TOKENS = int(os.getenv("FREE_DAILY_TOKENS", "100"))

MONTHLY_PLAN_TOKENS = {
    "plus": int(os.getenv("PLUS_MONTHLY_TOKENS", "2000")),
    "pro": int(os.getenv("PRO_MONTHLY_TOKENS", "8000")),
}

PLAN_CATALOG = {
    "free": {
        "price_usd": 0.0,
        "daily_tokens": FREE_DAILY_TOKENS,
        "monthly_tokens": 0,
        "features": ["basic_processing", "standard_queue"],
    },
    "plus": {
        "price_usd": 4.99,
        "daily_tokens": 0,
        "monthly_tokens": MONTHLY_PLAN_TOKENS["plus"],
        "features": ["hd_export", "faster_queue", "premium_models"],
    },
    "pro": {
        "price_usd": 11.99,
        "daily_tokens": 0,
        "monthly_tokens": MONTHLY_PLAN_TOKENS["pro"],
        "features": ["4k_export", "priority_queue", "all_premium_models", "batch_jobs"],
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _DbConn:
    def __init__(self):
        self._is_postgres = USING_POSTGRES
        if self._is_postgres:
            self._conn = psycopg2.connect(DATABASE_URL)
        else:
            self._conn = sqlite3.connect(DB_PATH)
            self._conn.row_factory = sqlite3.Row

    def _adapt_query(self, query: str) -> str:
        return query.replace("?", "%s") if self._is_postgres else query

    def execute(self, query: str, params: tuple | list = ()):
        if self._is_postgres:
            cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(self._adapt_query(query), tuple(params))
            return cursor
        return self._conn.execute(query, tuple(params))

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            try:
                self.rollback()
            finally:
                self.close()
            return False
        self.close()
        return False


def _connect() -> _DbConn:
    return _DbConn()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_token_ledger(
    conn: _DbConn,
    user_id: int,
    delta: int,
    reason: str,
    balance_after: int,
    metadata: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO token_ledger (user_id, delta, reason, metadata, balance_after, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, delta, reason, metadata, balance_after, _utc_now()),
    )


def _build_user_payload(row: Any) -> dict:
    now = datetime.now(timezone.utc)
    trial_ends_at = _parse_utc(row["trial_ends_at"])
    trial_active = bool(trial_ends_at and trial_ends_at > now)
    plan = row["plan"] or "free"
    billing_status = row["billing_status"] or "free"
    paid_active = billing_status == "active" and plan in MONTHLY_PLAN_TOKENS
    effective_plan = "trial" if trial_active else (plan if paid_active else "free")

    return {
        "id": row["id"],
        "email": row["email"],
        "username": row["username"],
        "tokens": row["tokens"],
        "plan": plan,
        "billing_status": billing_status,
        "effective_plan": effective_plan,
        "trial_started_at": row["trial_started_at"],
        "trial_ends_at": row["trial_ends_at"],
        "trial_active": trial_active,
        "has_premium": trial_active or paid_active,
        "youtube_connected": bool(row["youtube_token"]),
    }


def _is_daily_refresh_due(last_refresh_str: str | None, now_ts: datetime) -> bool:
    if not last_refresh_str:
        return True
    last_refresh = _parse_utc(last_refresh_str)
    if not last_refresh:
        return True
    return (now_ts - last_refresh).total_seconds() >= 86400


def _is_monthly_refresh_due(last_refresh_str: str | None, now_ts: datetime) -> bool:
    if not last_refresh_str:
        return True
    last_refresh = _parse_utc(last_refresh_str)
    if not last_refresh:
        return True
    return (now_ts - last_refresh).days >= 30


def _column_exists(conn: _DbConn, table_name: str, column_name: str) -> bool:
    if USING_POSTGRES:
        row = conn.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = ? AND column_name = ?
            LIMIT 1
            """,
            (table_name, column_name),
        ).fetchone()
        return bool(row)

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def _ensure_unique_username(conn: _DbConn, base: str) -> str:
    candidate = base
    suffix = 1
    while conn.execute(
        "SELECT 1 FROM users WHERE lower(username) = lower(?)",
        (candidate,),
    ).fetchone():
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate


def _sanitize_username(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isalnum() or ch in {"_", "-", "."})
    return cleaned.lower()[:40]


def init_db() -> None:
    with _connect() as conn:
        if USING_POSTGRES:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    tokens INTEGER NOT NULL DEFAULT 100,
                    plan TEXT NOT NULL DEFAULT 'free',
                    billing_status TEXT NOT NULL DEFAULT 'free',
                    trial_started_at TEXT,
                    trial_ends_at TEXT,
                    monthly_tokens_reset_at TEXT NOT NULL DEFAULT '',
                    youtube_token TEXT,
                    last_token_refresh TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    tokens_added INTEGER NOT NULL,
                    amount_usd DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL,
                    gateway_session_id TEXT UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS password_resets (
                    token TEXT PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    delta INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    metadata TEXT,
                    balance_after INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    owner_user_id BIGINT,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    tokens INTEGER NOT NULL DEFAULT 100,
                    plan TEXT NOT NULL DEFAULT 'free',
                    billing_status TEXT NOT NULL DEFAULT 'free',
                    trial_started_at TEXT,
                    trial_ends_at TEXT,
                    monthly_tokens_reset_at TEXT NOT NULL DEFAULT '',
                    youtube_token TEXT,
                    last_token_refresh TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    tokens_added INTEGER NOT NULL,
                    amount_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    gateway_session_id TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS password_resets (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    delta INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    metadata TEXT,
                    balance_after INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    owner_user_id INTEGER,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

        # SQLite migrations for older databases.
        if not _column_exists(conn, "users", "username"):
            conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
        if not _column_exists(conn, "users", "youtube_token"):
            conn.execute("ALTER TABLE users ADD COLUMN youtube_token TEXT")
        if not _column_exists(conn, "users", "last_token_refresh"):
            conn.execute("ALTER TABLE users ADD COLUMN last_token_refresh TEXT DEFAULT ''")
        if not _column_exists(conn, "users", "plan"):
            conn.execute("ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free'")
        if not _column_exists(conn, "users", "billing_status"):
            conn.execute("ALTER TABLE users ADD COLUMN billing_status TEXT DEFAULT 'free'")
        if not _column_exists(conn, "users", "trial_started_at"):
            conn.execute("ALTER TABLE users ADD COLUMN trial_started_at TEXT")
        if not _column_exists(conn, "users", "trial_ends_at"):
            conn.execute("ALTER TABLE users ADD COLUMN trial_ends_at TEXT")
        if not _column_exists(conn, "users", "monthly_tokens_reset_at"):
            conn.execute("ALTER TABLE users ADD COLUMN monthly_tokens_reset_at TEXT DEFAULT ''")

        if not _column_exists(conn, "video_jobs", "payload_json"):
            # The table may not exist yet on older databases; create it before migrating columns.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    owner_user_id INTEGER,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

        if USING_POSTGRES:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_nocase ON users ((lower(username)))"
            )
        else:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_nocase ON users(username COLLATE NOCASE)"
            )

        existing = conn.execute("SELECT id, email, username FROM users").fetchall()
        for row in existing:
            if row["username"]:
                continue
            email_local = row["email"].split("@", 1)[0]
            base = _sanitize_username(email_local) or f"user{row['id']}"
            username = _ensure_unique_username(conn, base)
            conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, row["id"]))

        # Backfill trial window and status for legacy users.
        legacy_users = conn.execute(
            """
            SELECT id, created_at, trial_started_at, trial_ends_at, plan, billing_status
            FROM users
            """
        ).fetchall()
        for row in legacy_users:
            created_at = row["created_at"] or _utc_now()
            trial_started_at = row["trial_started_at"] or created_at
            start_dt = _parse_utc(trial_started_at) or datetime.now(timezone.utc)
            policy_trial_end = (start_dt + timedelta(days=TRIAL_DAYS)).isoformat()
            current_trial_end = row["trial_ends_at"]

            # Keep legacy data, but cap longer historical trials to the current policy.
            if not current_trial_end:
                trial_ends_at = policy_trial_end
            else:
                parsed_current_end = _parse_utc(current_trial_end)
                if parsed_current_end and parsed_current_end > (start_dt + timedelta(days=TRIAL_DAYS)):
                    trial_ends_at = policy_trial_end
                else:
                    trial_ends_at = current_trial_end
            plan = row["plan"] or "free"
            billing_status = row["billing_status"] or "free"
            conn.execute(
                """
                UPDATE users
                SET trial_started_at = ?, trial_ends_at = ?, plan = ?, billing_status = ?
                WHERE id = ?
                """,
                (trial_started_at, trial_ends_at, plan, billing_status, row["id"]),
            )

        conn.commit()


def upsert_video_job(job_id: str, payload: dict, owner_user_id: int | None = None) -> None:
    status = str(payload.get("status") or "processing")
    payload_json = json.dumps(payload, ensure_ascii=False)
    with _connect() as conn:
        if USING_POSTGRES:
            conn.execute(
                """
                INSERT INTO video_jobs (job_id, payload_json, owner_user_id, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (job_id) DO UPDATE SET
                    payload_json = EXCLUDED.payload_json,
                    owner_user_id = EXCLUDED.owner_user_id,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
                """,
                (job_id, payload_json, owner_user_id, status, _utc_now()),
            )
        else:
            conn.execute(
                """
                INSERT INTO video_jobs (job_id, payload_json, owner_user_id, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    owner_user_id = excluded.owner_user_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (job_id, payload_json, owner_user_id, status, _utc_now()),
            )
        conn.commit()


def get_video_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM video_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload_json"])
    return payload if isinstance(payload, dict) else None


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return digest.hex()


def create_user(username: str, email: str, password: str, starting_tokens: int = 100) -> dict:
    normalized_username = _sanitize_username(username)
    if len(normalized_username) < 3:
        raise ValueError("Username must be at least 3 valid characters")

    normalized_email = email.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    now = datetime.now(timezone.utc)
    trial_ends = now + timedelta(days=TRIAL_DAYS)

    with _connect() as conn:
        insert_params = (
            normalized_email,
            normalized_username,
            password_hash,
            salt.hex(),
            starting_tokens,
            "free",
            "trial",
            now.isoformat(),
            trial_ends.isoformat(),
            now.isoformat(),
            now.isoformat(),
        )

        if USING_POSTGRES:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    email, username, password_hash, salt, tokens,
                    plan, billing_status, trial_started_at, trial_ends_at,
                    monthly_tokens_reset_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                insert_params,
            )
            inserted = cursor.fetchone()
        else:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    email, username, password_hash, salt, tokens,
                    plan, billing_status, trial_started_at, trial_ends_at,
                    monthly_tokens_reset_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_params,
            )
            inserted = {"id": cursor.lastrowid}

        conn.commit()
        if inserted and isinstance(inserted, dict):
            user_id = inserted["id"]
        elif inserted:
            user_id = inserted[0]
        else:
            raise ValueError("Failed to create user")

    return {
        "id": user_id,
        "email": normalized_email,
        "username": normalized_username,
        "tokens": starting_tokens,
    }


def authenticate_user(login: str, password: str) -> dict | None:
    normalized_login = login.strip().lower()
    with _connect() as conn:
        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE lower(email) = ? OR lower(username) = ?
            """,
            (normalized_login, normalized_login),
        ).fetchone()

    if not user:
        return None

    salt = bytes.fromhex(user["salt"])
    expected = _hash_password(password, salt)
    if expected != user["password_hash"]:
        return None

    return _build_user_payload(user)


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _utc_now()),
        )
        conn.commit()
    return token


def get_user_by_token(token: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                u.id,
                u.email,
                u.username,
                u.tokens,
                u.plan,
                u.billing_status,
                u.trial_started_at,
                u.trial_ends_at,
                u.youtube_token
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()

    if not row:
        return None

    return _build_user_payload(row)


def get_user_by_login(login: str) -> dict | None:
    normalized = login.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, email, username, tokens, plan, billing_status, trial_started_at, trial_ends_at, youtube_token
            FROM users
            WHERE lower(email) = ? OR lower(username) = ?
            """,
            (normalized, normalized),
        ).fetchone()
    if not row:
        return None
    return _build_user_payload(row)


def deduct_tokens(user_id: int, token_cost: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET tokens = tokens - ?
            WHERE id = ? AND tokens >= ?
            """,
            (token_cost, user_id, token_cost),
        )
        if cursor.rowcount > 0:
            row = conn.execute("SELECT tokens FROM users WHERE id = ?", (user_id,)).fetchone()
            balance = int(row["tokens"]) if row else 0
            _record_token_ledger(conn, user_id, -token_cost, "usage_charge", balance)
        conn.commit()
        return cursor.rowcount > 0

def ensure_token_allowance(user_id: int, daily_amount: int | None = None):
    now_ts = datetime.now(timezone.utc)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
                id,
                tokens,
                plan,
                billing_status,
                trial_ends_at,
                last_token_refresh,
                monthly_tokens_reset_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return

        trial_ends_at = _parse_utc(row["trial_ends_at"])
        trial_active = bool(trial_ends_at and trial_ends_at > now_ts)
        plan = row["plan"] or "free"
        billing_status = row["billing_status"] or "free"
        paid_active = billing_status == "active" and plan in MONTHLY_PLAN_TOKENS
        tokens_now = int(row["tokens"])

        if billing_status == "trial" and not trial_active:
            conn.execute("UPDATE users SET billing_status = 'free' WHERE id = ?", (user_id,))
            billing_status = "free"

        if trial_active:
            refill_floor = TRIAL_DAILY_TOKENS
            reason = "daily_trial_refill"
            due = _is_daily_refresh_due(row["last_token_refresh"], now_ts)
            if due:
                new_balance = max(tokens_now, refill_floor)
                if new_balance != tokens_now:
                    conn.execute(
                        "UPDATE users SET tokens = ?, last_token_refresh = ? WHERE id = ?",
                        (new_balance, now_ts.isoformat(), user_id),
                    )
                    _record_token_ledger(conn, user_id, new_balance - tokens_now, reason, new_balance)
                else:
                    conn.execute(
                        "UPDATE users SET last_token_refresh = ? WHERE id = ?",
                        (now_ts.isoformat(), user_id),
                    )
        elif paid_active:
            monthly_floor = MONTHLY_PLAN_TOKENS[plan]
            due = _is_monthly_refresh_due(row["monthly_tokens_reset_at"], now_ts)
            if due:
                new_balance = max(tokens_now, monthly_floor)
                if new_balance != tokens_now:
                    conn.execute(
                        "UPDATE users SET tokens = ?, monthly_tokens_reset_at = ? WHERE id = ?",
                        (new_balance, now_ts.isoformat(), user_id),
                    )
                    _record_token_ledger(conn, user_id, new_balance - tokens_now, f"monthly_{plan}_refill", new_balance)
                else:
                    conn.execute(
                        "UPDATE users SET monthly_tokens_reset_at = ? WHERE id = ?",
                        (now_ts.isoformat(), user_id),
                    )
        else:
            refill_floor = int(daily_amount if daily_amount is not None else FREE_DAILY_TOKENS)
            due = _is_daily_refresh_due(row["last_token_refresh"], now_ts)
            if due:
                new_balance = max(tokens_now, refill_floor)
                if new_balance != tokens_now:
                    conn.execute(
                        "UPDATE users SET tokens = ?, last_token_refresh = ? WHERE id = ?",
                        (new_balance, now_ts.isoformat(), user_id),
                    )
                    _record_token_ledger(conn, user_id, new_balance - tokens_now, "daily_free_refill", new_balance)
                else:
                    conn.execute(
                        "UPDATE users SET last_token_refresh = ? WHERE id = ?",
                        (now_ts.isoformat(), user_id),
                    )

        conn.commit()


def add_tokens(
    user_id: int,
    token_count: int,
    amount_usd: float = 0.0,
    status: str = "paid",
    gateway_session_id: str | None = None,
) -> int:
    with _connect() as conn:
        if gateway_session_id:
            existing = conn.execute(
                "SELECT 1 FROM payments WHERE gateway_session_id = ?",
                (gateway_session_id,),
            ).fetchone()
            if existing:
                row = conn.execute("SELECT tokens FROM users WHERE id = ?", (user_id,)).fetchone()
                return int(row["tokens"]) if row else 0

        conn.execute("UPDATE users SET tokens = tokens + ? WHERE id = ?", (token_count, user_id))
        row = conn.execute("SELECT tokens FROM users WHERE id = ?", (user_id,)).fetchone()
        balance = int(row["tokens"]) if row else 0
        _record_token_ledger(conn, user_id, token_count, status, balance, metadata=gateway_session_id)
        conn.execute(
            """
            INSERT INTO payments (user_id, tokens_added, amount_usd, status, gateway_session_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, token_count, amount_usd, status, gateway_session_id, _utc_now()),
        )
        conn.commit()
        return balance


def refund_tokens(user_id: int, token_count: int) -> int:
    return add_tokens(user_id, token_count, amount_usd=0.0, status="refunded")


def create_password_reset_token(user_id: int, expires_minutes: int = 30) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(minutes=expires_minutes)).isoformat()

    with _connect() as conn:
        conn.execute("DELETE FROM password_resets WHERE user_id = ? AND used_at IS NULL", (user_id,))
        conn.execute(
            """
            INSERT INTO password_resets (token, user_id, expires_at, used_at, created_at)
            VALUES (?, ?, ?, NULL, ?)
            """,
            (token, user_id, expires_at, now.isoformat()),
        )
        conn.commit()

    return token


def consume_password_reset_token(token: str, new_password: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT token, user_id, expires_at, used_at
            FROM password_resets
            WHERE token = ?
            """,
            (token,),
        ).fetchone()

        if not row or row["used_at"]:
            return False

        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
        except ValueError:
            return False

        if expires_at < datetime.now(timezone.utc):
            return False

        salt = secrets.token_bytes(16)
        password_hash = _hash_password(new_password, salt)
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (password_hash, salt.hex(), row["user_id"]),
        )
        conn.execute(
            "UPDATE password_resets SET used_at = ? WHERE token = ?",
            (_utc_now(), token),
        )
        conn.commit()
        return True

def save_youtube_token(user_id: int, token_json: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET youtube_token = ? WHERE id = ?", (token_json, user_id))
        conn.commit()

def get_youtube_token(user_id: int) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT youtube_token FROM users WHERE id = ?", (user_id,)).fetchone()
        return row["youtube_token"] if row else None


def get_plan_catalog() -> dict:
    return {
        "trial_days": TRIAL_DAYS,
        "trial_daily_tokens": TRIAL_DAILY_TOKENS,
        "plans": PLAN_CATALOG,
    }


def set_subscription_plan(user_id: int, plan: str, active: bool = True) -> dict:
    normalized_plan = (plan or "").strip().lower()
    if normalized_plan not in {"free", "plus", "pro"}:
        raise ValueError("Invalid plan")

    new_status = "active" if active and normalized_plan in {"plus", "pro"} else "free"
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE users
            SET plan = ?, billing_status = ?, monthly_tokens_reset_at = CASE WHEN ? = 'active' THEN '' ELSE monthly_tokens_reset_at END
            WHERE id = ?
            """,
            (normalized_plan, new_status, new_status, user_id),
        )
        conn.commit()

    ensure_token_allowance(user_id)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, email, username, tokens, plan, billing_status, trial_started_at, trial_ends_at, youtube_token
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    if not row:
        raise ValueError("User not found")
    return _build_user_payload(row)
