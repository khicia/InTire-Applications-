import os
import random
import cv2
import numpy as np
from datetime import date, datetime, timedelta

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash


# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder=".")

# Mail
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="your@gmail.com",
    MAIL_PASSWORD="generated gmail app password",
)
mail = Mail(app)

# Session
app.config.update(
    SECRET_KEY=os.environ.get("INTIRE_SECRET_KEY", "dev-secret-change-me"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

CORS(app, supports_credentials=True)

# Database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, "instance", "intire.db")
os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)

app.config.update(
    SQLALCHEMY_DATABASE_URI="sqlite:///" + db_path,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

db = SQLAlchemy(app)
otp_storage: dict[str, str] = {}


# ─────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────

class Role(db.Model):
    __tablename__ = "role_tbl"
    role_code = db.Column(db.Integer, primary_key=True)
    role_desc = db.Column(db.String(20))


class Account(db.Model):
    __tablename__ = "account_tbl"
    account_no = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fname      = db.Column(db.String(50))
    mname      = db.Column(db.String(50))
    lname      = db.Column(db.String(50))
    mobile_no  = db.Column(db.String(20))
    email      = db.Column(db.String(100), unique=True)
    password   = db.Column(db.String(255))
    role_code  = db.Column(db.Integer, db.ForeignKey("role_tbl.role_code"), default=1)


class Inspection(db.Model):
    __tablename__ = "inspection_tbl"
    inspection_no   = db.Column(db.Integer, primary_key=True, autoincrement=True)
    plate_no        = db.Column(db.String(20))
    inspection_date = db.Column(db.String(20))
    vehicle_type    = db.Column(db.String(50))
    vehicle_model   = db.Column(db.String(100))
    inspector       = db.Column(db.String(100))
    date_inspected  = db.Column(db.String(20))


class DefectType(db.Model):
    __tablename__ = "defect_tbl"
    defect_code = db.Column(db.Integer, primary_key=True)
    defect_type = db.Column(db.String(50))


class TireDefect(db.Model):
    __tablename__ = "tiredefects_tbl"
    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    plate_no         = db.Column(db.String(20), db.ForeignKey("inspection_tbl.plate_no"))
    tire_position    = db.Column(db.String(30))
    tire_dot         = db.Column(db.String(4))
    manufacture_date = db.Column(db.String(50))
    expiry_date      = db.Column(db.String(50))
    tire_age         = db.Column(db.String(20))
    validity         = db.Column(db.String(20))
    defect_code      = db.Column(db.Integer, db.ForeignKey("defect_tbl.defect_code"))
    image_data       = db.Column(db.Text)


class Notification(db.Model):
    __tablename__ = "notification_tbl"
    notification_no      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    notification_content = db.Column(db.Text)
    notification_type    = db.Column(db.Integer, db.ForeignKey("notification_type_tbl.notification_type"))


class NotificationType(db.Model):
    __tablename__ = "notification_type_tbl"
    notification_type = db.Column(db.Integer, primary_key=True)
    notification_desc = db.Column(db.String(20))


# ─────────────────────────────────────────────
#  SEED & INIT
# ─────────────────────────────────────────────

with app.app_context():
    db.create_all()

    if not Role.query.first():
        db.session.add_all([
            Role(role_code=1, role_desc="user"),
            Role(role_code=2, role_desc="admin"),
            Role(role_code=3, role_desc="guest"),
        ])

    if not DefectType.query.first():
        db.session.add_all([
            DefectType(defect_code=1, defect_type="Surface Crack"),
            DefectType(defect_code=2, defect_type="Bulge"),
            DefectType(defect_code=3, defect_type="Worn Tread"),
            DefectType(defect_code=4, defect_type="Puncture Hole"),
            DefectType(defect_code=5, defect_type="Puncture Object"),
        ])

    if not NotificationType.query.first():
        db.session.add_all([
            NotificationType(notification_type=1, notification_desc="alerts"),
            NotificationType(notification_type=2, notification_desc="info"),
            NotificationType(notification_type=3, notification_desc="system"),
        ])

    db.session.commit()

model = YOLO(r"assets\ai model\best.pt")


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _parse_date(s: str) -> date | None:
    """Parse a YYYY-MM-DD string, returning None on failure."""
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _vehicle_title(insp: Inspection) -> str:
    parts = [p for p in [(insp.vehicle_model or "").strip(), (insp.plate_no or "").strip()] if p]
    return " · ".join(parts) or f"Inspection #{insp.inspection_no}"


def _active_plate_nos(plate_nos: list[str]) -> set[str]:
    if not plate_nos:
        return set()
    return set(
        r[0] for r in db.session.query(TireDefect.plate_no)
        .filter(TireDefect.plate_no.in_(plate_nos))
        .distinct()
        .all()
    )


# ─────────────────────────────────────────────
#  STATIC FILES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def serve_file(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    return "File not found", 404


# ─────────────────────────────────────────────
#  AI PREDICTION
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    img = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    detections = [
        {
            "class": int(b.cls[0]),
            "confidence": float(b.conf[0]),
            "bbox": b.xyxy[0].tolist(),
        }
        for r in model(img)
        for b in r.boxes
    ]
    return jsonify(detections)


# ─────────────────────────────────────────────
#  INSPECTIONS
# ─────────────────────────────────────────────

@app.route("/save-inspection", methods=["POST"])
def save_inspection():
    data = request.get_json()
    inspection_no = data.get("inspectionNo")

    if not inspection_no:
        inspection = Inspection(
            plate_no        = data.get("plateNumber"),
            inspection_date = data.get("date"),
            vehicle_type    = data.get("carType"),
            vehicle_model   = data.get("vehicleModel"),
            inspector       = data.get("inspectedBy"),
            date_inspected  = data.get("date"),
        )
        db.session.add(inspection)
        db.session.flush()
        inspection_no = inspection.inspection_no

        db.session.add(Notification(
            notification_content=f"Inspection completed for plate {data.get('plateNumber')}",
            notification_type=2,
        ))
    else:
        inspection = db.session.get(Inspection, inspection_no)
        if not inspection:
            return jsonify({"success": False, "error": "Inspection not found"}), 404

    plate_no = data.get("plateNumber") or inspection.plate_no
    image_data = data.get("imageBase64", "")

    for defect_code in data.get("defects", []):
        db.session.add(TireDefect(
            plate_no         = plate_no,
            tire_position    = data.get("tirePosition"),
            tire_dot         = data.get("dotCode"),
            manufacture_date = data.get("manufactureDate"),
            expiry_date      = data.get("expiryDate"),
            tire_age         = data.get("tireAge"),
            validity         = data.get("validity"),
            defect_code      = defect_code,
            image_data       = image_data,
        ))

    db.session.commit()
    return jsonify({"success": True, "inspection_no": inspection_no})


@app.route("/inspection-history", methods=["GET"])
def get_inspection_history():
    inspections = Inspection.query.order_by(Inspection.inspection_no.desc()).all()
    result = []

    for insp in inspections:
        tires = TireDefect.query.filter_by(plate_no=insp.plate_no).all()
        defect_list = []
        for t in tires:
            defect = db.session.get(DefectType, t.defect_code)
            defect_list.append({
                "tirePosition":    t.tire_position,
                "dotCode":         t.tire_dot,
                "manufactureDate": t.manufacture_date,
                "expiryDate":      t.expiry_date,
                "tireAge":         t.tire_age,
                "validity":        t.validity,
                "defectType":      defect.defect_type if defect else "Unknown",
                "imageData":       t.image_data or "",
            })

        result.append({
            "inspectionNo":  insp.inspection_no,
            "plateNo":       insp.plate_no,
            "vehicleType":   insp.vehicle_type,
            "vehicleModel":  insp.vehicle_model,
            "inspector":     insp.inspector,
            "dateInspected": insp.date_inspected,
            "tires":         defect_list,
        })

    return jsonify(result)


# ─────────────────────────────────────────────
#  DEFECT TYPES
# ─────────────────────────────────────────────

@app.route("/defects", methods=["GET"])
def get_defects():
    return jsonify([
        {"defectCode": d.defect_code, "defectType": d.defect_type}
        for d in DefectType.query.all()
    ])


# ─────────────────────────────────────────────
#  NOTIFICATIONS
# ─────────────────────────────────────────────

@app.route("/notifications", methods=["GET"])
def get_notifications():
    return jsonify([
        {
            "notificationNo":      n.notification_no,
            "notificationContent": n.notification_content,
            "notificationType":    n.notification_type,
        }
        for n in Notification.query.order_by(Notification.notification_no.desc()).all()
    ])


# ─────────────────────────────────────────────
#  ANALYTICS
# ─────────────────────────────────────────────

def _build_period(period: str, today: date) -> tuple:
    """Return (start, end, labels, bars, bucket_fn) for a given period."""
    if period == "year":
        start = date(today.year, 1, 1)
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        bars = [0] * 12
        bucket_fn = lambda d: d.month - 1

    elif period == "month":
        start = today - timedelta(days=27)
        labels = ["Week 1", "Week 2", "Week 3", "Week 4"]
        bars = [0] * 4
        bucket_fn = lambda d: max(0, min(3, (d - start).days // 7))

    else:  # week (default)
        start = today - timedelta(days=6)
        labels = [(start + timedelta(days=i)).strftime("%a") for i in range(7)]
        bars = [0] * 7
        bucket_fn = lambda d: (d - start).days

    return start, today, labels, bars, bucket_fn


@app.route("/analytics", methods=["GET"])
def analytics():
    period = (request.args.get("period") or "week").lower()
    today = date.today()
    start, end, labels, bars, bucket_fn = _build_period(period, today)

    # Filter inspections within range
    inspections_in_range = [
        (insp, d)
        for insp in Inspection.query.all()
        if (d := _parse_date(insp.date_inspected or insp.inspection_date)) and start <= d <= end
    ]

    for _, d in inspections_in_range:
        idx = bucket_fn(d)
        if 0 <= idx < len(bars):
            bars[idx] += 1

    total = len(inspections_in_range)
    plate_nos = [insp.plate_no for insp, _ in inspections_in_range if insp.plate_no]
    active_plates = _active_plate_nos(plate_nos)
    active = sum(1 for insp, _ in inspections_in_range if insp.plate_no in active_plates)
    resolved = max(0, total - active)
    rate = f"{int(round(resolved / total * 100))}%" if total else "0%"

    # Defect breakdown
    breakdown = []
    if plate_nos:
        rows = (
            db.session.query(DefectType.defect_type, func.count(TireDefect.id))
            .join(TireDefect, TireDefect.defect_code == DefectType.defect_code)
            .filter(TireDefect.plate_no.in_(set(plate_nos)))
            .group_by(DefectType.defect_type)
            .order_by(func.count(TireDefect.id).desc())
            .all()
        )
        breakdown = [{"name": r[0], "count": int(r[1])} for r in rows]
    breakdown.append({"name": "No Damage", "count": int(resolved)})

    # Top vehicles
    vehicle_counts: dict[str, int] = {}
    for insp, _ in inspections_in_range:
        title = _vehicle_title(insp)
        vehicle_counts[title] = vehicle_counts.get(title, 0) + 1
    vehicles = [
        {"name": n, "count": c}
        for n, c in sorted(vehicle_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
    ]

    label_map = {"week": "This Week", "month": "This Month", "year": "This Year"}
    return jsonify({
        "period": period,
        "label": label_map.get(period, "This Week"),
        "bars": bars,
        "barLabels": labels,
        "stats": {"total": total, "active": active, "resolved": resolved, "rate": rate},
        "breakdown": breakdown,
        "vehicles": vehicles,
    })


# ─────────────────────────────────────────────
#  ADMIN DASHBOARD
# ─────────────────────────────────────────────

@app.route("/admin-dashboard", methods=["GET"])
def admin_dashboard():
    today = date.today()
    week_start = today - timedelta(days=6)

    all_inspections = Inspection.query.all()
    total_inspections = len(all_inspections)
    plate_nos = [i.plate_no for i in all_inspections if i.plate_no]
    active_plates = _active_plate_nos(plate_nos)
    active_issues = sum(1 for i in all_inspections if i.plate_no in active_plates)
    resolved = max(0, total_inspections - active_issues)
    resolution_rate = resolved / total_inspections if total_inspections else 0.0

    # Weekly bar chart
    week_labels = [(week_start + timedelta(days=i)).strftime("%a") for i in range(7)]
    week_bars = [0] * 7
    inspections_dated = []

    for insp in all_inspections:
        d = _parse_date(insp.date_inspected or insp.inspection_date)
        if not d:
            continue
        inspections_dated.append((insp, d))
        if week_start <= d <= today:
            week_bars[(d - week_start).days] += 1

    # Most active inspectors (top 5)
    inspector_counts: dict[str, int] = {}
    for insp, _ in inspections_dated:
        name = (insp.inspector or "").strip()
        if name:
            inspector_counts[name] = inspector_counts.get(name, 0) + 1

    most_active_users = [
        {
            "name": name,
            "email": "",
            "initials": "".join(p[0].upper() for p in name.split()[:2] if p)[:2] or "U",
            "count": int(count),
        }
        for name, count in sorted(inspector_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]

    # All-time defect breakdown
    rows = (
        db.session.query(DefectType.defect_type, func.count(TireDefect.id))
        .join(TireDefect, TireDefect.defect_code == DefectType.defect_code)
        .group_by(DefectType.defect_type)
        .order_by(func.count(TireDefect.id).desc())
        .all()
    )
    breakdown = [{"name": r[0], "count": int(r[1])} for r in rows]
    breakdown.append({"name": "No Damage", "count": int(resolved)})

    # Recent inspections (latest 8)
    recent = []
    for insp in Inspection.query.order_by(Inspection.inspection_no.desc()).limit(8).all():
        tires = TireDefect.query.filter_by(plate_no=insp.plate_no).all() if insp.plate_no else []
        if tires:
            defect = db.session.get(DefectType, tires[0].defect_code)
            defect_name = defect.defect_type if defect else "Unknown"
            position = tires[0].tire_position or ""
            status = "active"
        else:
            defect_name, position, status = "No Damage", "", "resolved"

        recent.append({
            "inspectionNo": insp.inspection_no,
            "vehicle":      _vehicle_title(insp),
            "user":         (insp.inspector or "").strip(),
            "position":     position,
            "defect":       defect_name,
            "date":         (insp.date_inspected or insp.inspection_date or "").strip(),
            "status":       status,
        })

    return jsonify({
        "stats": {
            "totalUsers":       int(Account.query.count()),
            "totalInspections": int(total_inspections),
            "activeIssues":     int(active_issues),
            "resolutionRate":   f"{int(round(resolution_rate * 100))}%",
        },
        "week":             {"bars": week_bars, "labels": week_labels},
        "mostActiveUsers":  most_active_users,
        "breakdown":        breakdown,
        "recentInspections": recent,
    })


# ─────────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────────

@app.route("/sign-up", methods=["POST"])
def sign_up():
    data = request.get_json() or {}
    fname     = (data.get("fname") or "").strip()
    mname     = (data.get("mname") or "").strip()
    lname     = (data.get("lname") or "").strip()
    mobile_no = (data.get("mobileNo") or "").strip()
    email     = (data.get("email") or "").strip().lower()
    password  = data.get("password") or ""

    if not all([fname, lname, mobile_no, email, password]):
        return jsonify({"success": False, "error": "Please fill in all required fields."}), 400
    if "@" not in email or "." not in email:
        return jsonify({"success": False, "error": "Please enter a valid email address."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400
    if Account.query.filter_by(email=email).first():
        return jsonify({"success": False, "error": "Email already registered."}), 409

    acc = Account(
        fname=fname, mname=mname, lname=lname,
        mobile_no=mobile_no, email=email,
        password=generate_password_hash(password),
        role_code=1,
    )
    db.session.add(acc)
    db.session.commit()
    return jsonify({"success": True, "accountNo": acc.account_no})


@app.route("/sign-in", methods=["POST"])
def sign_in():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "error": "Please enter email and password."}), 400

    acc = Account.query.filter_by(email=email).first()
    if not acc or not check_password_hash(acc.password, password):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    session.permanent = True
    session["account_no"] = acc.account_no
    return jsonify({
        "success":  True,
        "accountNo": acc.account_no,
        "fname":    acc.fname,
        "mname":    acc.mname,
        "lname":    acc.lname,
        "email":    acc.email,
        "mobileNo": acc.mobile_no,
        "roleCode": acc.role_code,
    })


@app.route("/me", methods=["GET"])
def me():
    if session.get("is_guest"):
        return jsonify({"success": True, "isGuest": True, "roleCode": 3})

    account_no = session.get("account_no")
    if not account_no:
        return jsonify({"success": False, "error": "Not logged in."}), 401

    acc = db.session.get(Account, account_no)
    if not acc:
        session.pop("account_no", None)
        return jsonify({"success": False, "error": "Not logged in."}), 401

    return jsonify({
        "success":   True,
        "accountNo": acc.account_no,
        "fname":     acc.fname,
        "mname":     acc.mname,
        "lname":     acc.lname,
        "email":     acc.email,
        "mobileNo":  acc.mobile_no,
        "roleCode":  acc.role_code,
    })


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("account_no", None)
    session.pop("is_guest", None)
    return jsonify({"success": True})


@app.route("/guest-session", methods=["POST"])
def guest_session():
    session.clear()
    session.permanent = True
    session["is_guest"] = True
    return jsonify({"success": True, "isGuest": True})


# ─────────────────────────────────────────────
#  PASSWORD RESET
# ─────────────────────────────────────────────

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    data       = request.get_json() or {}
    identifier = (data.get("identifier") or "").strip()

    if not identifier:
        return jsonify({"success": False, "error": "Please enter your email or phone number."}), 400

    acc = Account.query.filter(
        (Account.email == identifier.lower()) | (Account.mobile_no == identifier)
    ).first()

    if not acc:
        return jsonify({"success": False, "error": "Account not found."}), 404

    otp = str(random.randint(100000, 999999))
    otp_storage[acc.email] = otp

    msg = Message(
        "InTire Password Reset OTP",
        sender=app.config["MAIL_USERNAME"],
        recipients=[acc.email],
    )
    msg.body = f"Your OTP code is: {otp}\n\nDo not share this code with anyone."
    mail.send(msg)

    return jsonify({"success": True, "message": "OTP sent successfully.", "email": acc.email})


@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data  = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    otp   = (data.get("otp") or "").strip()

    saved_otp = otp_storage.get(email)
    if not saved_otp:
        return jsonify({"success": False, "error": "OTP expired."}), 400
    if saved_otp != otp:
        return jsonify({"success": False, "error": "Invalid OTP."}), 400

    return jsonify({"success": True, "message": "OTP verified."})


@app.route("/reset-password", methods=["POST"])
def reset_password():
    data         = request.get_json() or {}
    email        = (data.get("email") or "").strip().lower()
    new_password = (data.get("password") or "").strip()

    if len(new_password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400

    acc = Account.query.filter_by(email=email).first()
    if not acc:
        return jsonify({"success": False, "error": "Account not found."}), 404

    acc.password = generate_password_hash(new_password)
    db.session.commit()
    otp_storage.pop(email, None)

    return jsonify({"success": True, "message": "Password updated successfully."})


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)