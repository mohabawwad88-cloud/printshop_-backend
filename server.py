from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import uuid
from datetime import datetime
import base64
from PIL import Image
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection with error handling
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'hm_planet')

try:
    client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
    db = client[db_name]
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# CORS configuration from environment
cors_origins = os.environ.get('CORS_ORIGINS', '*')
if cors_origins == '*':
    cors_origins_list = ["*"]
else:
    cors_origins_list = [origin.strip() for origin in cors_origins.split(',')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ MODELS ============

class UserLogin(BaseModel):
    phone: str

class User(BaseModel):
    phone: str
    role: Literal['user', 'admin'] = 'user'
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Product(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    price: float
    image_base64: str
    category: Literal['stationery', 'teaching_notes', 'gifts', 'toys']
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ProductCreate(BaseModel):
    name: str
    description: str
    price: float
    image_base64: str
    category: Literal['stationery', 'teaching_notes', 'gifts', 'toys']

class PrintOptions(BaseModel):
    paper_size: str  # A4, A3, A5
    color_mode: Literal['black', 'color']
    sides: Literal['single', 'double']
    binding: bool = False
    stapling: bool = False

class PrintOrder(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_phone: str
    file_base64: str
    file_name: str
    file_type: str  # 'document' or 'image'
    options: PrintOptions
    total_pages: int
    total_cost: float
    status: Literal['received', 'reviewing', 'printing', 'ready'] = 'received'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PrintOrderCreate(BaseModel):
    user_phone: str
    file_base64: str
    file_name: str
    file_type: str
    options: PrintOptions
    total_pages: int

class PrintOrderUpdate(BaseModel):
    status: Literal['received', 'reviewing', 'printing', 'ready']

class CostCalculation(BaseModel):
    total_pages: int
    options: PrintOptions

# Shopping order models
class OrderItem(BaseModel):
    product_id: str
    name: str
    price: float
    quantity: int

class ShoppingOrder(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_phone: str
    items: List[OrderItem]
    total_cost: float
    status: Literal['pending', 'confirmed', 'ready', 'completed'] = 'pending'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ShoppingOrderCreate(BaseModel):
    user_phone: str
    items: List[OrderItem]
    total_cost: float

class ShoppingOrderUpdate(BaseModel):
    status: Literal['pending', 'confirmed', 'ready', 'completed']

# Offer/Discount models
class Offer(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    image_base64: str
    discount_percentage: Optional[float] = None
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class OfferCreate(BaseModel):
    title: str
    description: str
    image_base64: str
    discount_percentage: Optional[float] = None
    end_date: Optional[datetime] = None

# ============ AUTH ROUTES ============

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    phone = user_data.phone.strip()
    
    # التحقق المبسط - يجب أن يبدأ بـ 09 وأن يكون طوله مناسب
    if not phone.startswith('09'):
        raise HTTPException(status_code=400, detail="رقم الهاتف يجب أن يبدأ بـ 09")
    
    if len(phone) < 10 or len(phone) > 12:
        raise HTTPException(status_code=400, detail="رقم الهاتف غير صالح")
    
    # التحقق مما إذا كان المستخدم موجوداً
    user = await db.users.find_one({"phone": phone})
    
    if not user:
        # تحديد إذا كان مدير (ينتهي بـ ++)
        role = 'admin' if phone.endswith('++') else 'user'
        
        # إذا كان مدير، تحقق من الحد الأقصى
        if role == 'admin':
            admin_count = await db.users.count_documents({"role": "admin"})
            if admin_count >= 3:
                raise HTTPException(status_code=403, detail="تم الوصول للحد الأقصى من المديرين (3)")
        
        # إنشاء مستخدم جديد تلقائياً
        new_user = User(phone=phone, role=role)
        await db.users.insert_one(new_user.dict())
        logger.info(f"New user created: {phone}, role: {role}")
        return {"phone": phone, "role": role, "message": "تم تسجيل الدخول بنجاح", "is_new": True}
    else:
        return {"phone": user['phone'], "role": user['role'], "message": "مرحباً بعودتك", "is_new": False}

# ============ PRODUCT ROUTES ============

@api_router.get("/products/{category}")
async def get_products(category: str, page: int = 1, limit: int = 50):
    if category not in ['stationery', 'teaching_notes', 'gifts', 'toys', 'all']:
        raise HTTPException(status_code=400, detail="قسم غير صحيح")
    
    # Ensure valid pagination
    page = max(1, page)
    limit = min(100, max(1, limit))
    skip = (page - 1) * limit
    
    query = {} if category == 'all' else {"category": category}
    products = await db.products.find(query).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.products.count_documents(query)
    
    # Convert ObjectId to string for JSON serialization
    for product in products:
        if '_id' in product:
            product['_id'] = str(product['_id'])
    
    return {
        "products": products,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.post("/products", response_model=Product)
async def create_product(product: ProductCreate, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    product_obj = Product(**product.dict())
    await db.products.insert_one(product_obj.dict())
    return product_obj

@api_router.put("/products/{product_id}")
async def update_product(product_id: str, product: ProductCreate, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    result = await db.products.update_one(
        {"id": product_id},
        {"$set": product.dict()}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="المنتج غير موجود")
    
    return {"message": "تم تحديث المنتج بنجاح"}

@api_router.delete("/products/{product_id}")
async def delete_product(product_id: str, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    result = await db.products.delete_one({"id": product_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="المنتج غير موجود")
    
    return {"message": "تم حذف المنتج بنجاح"}

# ============ PRINT ROUTES ============

@api_router.post("/print/calculate-cost")
async def calculate_cost(calculation: CostCalculation):
    price_per_page = 20 if calculation.options.color_mode == 'color' else 10
    
    # Calculate effective pages (double-sided prints half pages)
    effective_pages = calculation.total_pages
    if calculation.options.sides == 'double':
        effective_pages = (calculation.total_pages + 1) // 2
    
    total_cost = effective_pages * price_per_page
    
    # Add binding cost
    if calculation.options.binding:
        total_cost += 50  # 50 ل.س للتجليد
    
    # Add stapling cost
    if calculation.options.stapling:
        total_cost += 20  # 20 ل.س للتسليك
    
    return {
        "total_pages": calculation.total_pages,
        "effective_pages": effective_pages,
        "price_per_page": price_per_page,
        "printing_cost": effective_pages * price_per_page,
        "binding_cost": 50 if calculation.options.binding else 0,
        "stapling_cost": 20 if calculation.options.stapling else 0,
        "total_cost": total_cost
    }

@api_router.post("/print/orders", response_model=PrintOrder)
async def create_print_order(order: PrintOrderCreate):
    # Calculate cost
    price_per_page = 20 if order.options.color_mode == 'color' else 10
    effective_pages = order.total_pages
    if order.options.sides == 'double':
        effective_pages = (order.total_pages + 1) // 2
    
    total_cost = effective_pages * price_per_page
    if order.options.binding:
        total_cost += 50
    if order.options.stapling:
        total_cost += 20
    
    order_obj = PrintOrder(
        **order.dict(),
        total_cost=total_cost
    )
    
    await db.print_orders.insert_one(order_obj.dict())
    return order_obj

@api_router.get("/print/orders/user/{phone}")
async def get_user_orders(phone: str, page: int = 1, limit: int = 20):
    page = max(1, page)
    limit = min(50, max(1, limit))
    skip = (page - 1) * limit
    
    orders = await db.print_orders.find({"user_phone": phone}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.print_orders.count_documents({"user_phone": phone})
    
    # Convert ObjectId to string
    for order in orders:
        if '_id' in order:
            order['_id'] = str(order['_id'])
    
    return {
        "orders": orders,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.get("/print/orders/all")
async def get_all_orders(admin_phone: str, page: int = 1, limit: int = 50):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    page = max(1, page)
    limit = min(100, max(1, limit))
    skip = (page - 1) * limit
    
    orders = await db.print_orders.find().sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.print_orders.count_documents({})
    
    # Convert ObjectId to string
    for order in orders:
        if '_id' in order:
            order['_id'] = str(order['_id'])
    
    return {
        "orders": orders,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.put("/print/orders/{order_id}/status")
async def update_order_status(order_id: str, update: PrintOrderUpdate, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    result = await db.print_orders.update_one(
        {"id": order_id},
        {"$set": {"status": update.status, "updated_at": datetime.utcnow()}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="الطلب غير موجود")
    
    # Get updated order
    order = await db.print_orders.find_one({"id": order_id})
    
    return {" message": "تم تحديث حالة الطلب بنجاح", "order": order}

# ============ SHOPPING ORDER ROUTES ============

@api_router.post("/orders", response_model=ShoppingOrder)
async def create_shopping_order(order: ShoppingOrderCreate):
    order_obj = ShoppingOrder(**order.dict())
    await db.shopping_orders.insert_one(order_obj.dict())
    return order_obj

@api_router.get("/orders/user/{phone}")
async def get_user_shopping_orders(phone: str, page: int = 1, limit: int = 20):
    page = max(1, page)
    limit = min(50, max(1, limit))
    skip = (page - 1) * limit
    
    orders = await db.shopping_orders.find({"user_phone": phone}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.shopping_orders.count_documents({"user_phone": phone})
    
    # Convert ObjectId to string
    for order in orders:
        if '_id' in order:
            order['_id'] = str(order['_id'])
    
    return {
        "orders": orders,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.get("/orders/all")
async def get_all_shopping_orders(admin_phone: str, page: int = 1, limit: int = 50):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    page = max(1, page)
    limit = min(100, max(1, limit))
    skip = (page - 1) * limit
    
    orders = await db.shopping_orders.find().sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.shopping_orders.count_documents({})
    
    # Convert ObjectId to string
    for order in orders:
        if '_id' in order:
            order['_id'] = str(order['_id'])
    
    return {
        "orders": orders,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.put("/orders/{order_id}/status")
async def update_shopping_order_status(order_id: str, update: ShoppingOrderUpdate, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    result = await db.shopping_orders.update_one(
        {"id": order_id},
        {"$set": {"status": update.status, "updated_at": datetime.utcnow()}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="الطلب غير موجود")
    
    # Get updated order
    order = await db.shopping_orders.find_one({"id": order_id})
    
    return {"message": "تم تحديث حالة الطلب بنجاح", "order": order}

# ============ OFFERS/DISCOUNTS ROUTES ============

@api_router.get("/offers")
async def get_offers(page: int = 1, limit: int = 30):
    page = max(1, page)
    limit = min(50, max(1, limit))
    skip = (page - 1) * limit
    
    offers = await db.offers.find({"is_active": True}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    total = await db.offers.count_documents({"is_active": True})
    
    # Convert ObjectId to string
    for offer in offers:
        if '_id' in offer:
            offer['_id'] = str(offer['_id'])
    
    return {
        "offers": offers,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit
        }
    }

@api_router.post("/offers", response_model=Offer)
async def create_offer(offer: OfferCreate, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    offer_obj = Offer(**offer.dict())
    await db.offers.insert_one(offer_obj.dict())
    
    # Here you would trigger push notifications to all users
    # For now, we'll just return the offer
    
    return offer_obj

@api_router.delete("/offers/{offer_id}")
async def delete_offer(offer_id: str, admin_phone: str):
    # Verify admin
    user = await db.users.find_one({"phone": admin_phone})
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="غير مسموح")
    
    result = await db.offers.delete_one({"id": offer_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="العرض غير موجود")
    
    return {"message": "تم حذف العرض بنجاح"}

# ============ GENERAL ROUTES ============

@api_router.get("/")
async def root():
    return {"message": "HM Planet API"}

@api_router.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    try:
        # Test MongoDB connection
        await client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@app.on_event("startup")
async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Users indexes
        await db.users.create_index("phone", unique=True)
        await db.users.create_index("role")
        
        # Products indexes
        await db.products.create_index("id", unique=True)
        await db.products.create_index("category")
        await db.products.create_index("created_at")
        await db.products.create_index([("category", 1), ("created_at", -1)])
        
        # Print orders indexes
        await db.print_orders.create_index("id", unique=True)
        await db.print_orders.create_index("user_phone")
        await db.print_orders.create_index("status")
        await db.print_orders.create_index("created_at")
        await db.print_orders.create_index([("user_phone", 1), ("created_at", -1)])
        
        # Shopping orders indexes
        await db.shopping_orders.create_index("id", unique=True)
        await db.shopping_orders.create_index("user_phone")
        await db.shopping_orders.create_index("status")
        await db.shopping_orders.create_index("created_at")
        await db.shopping_orders.create_index([("user_phone", 1), ("created_at", -1)])
        
        # Offers indexes
        await db.offers.create_index("id", unique=True)
        await db.offers.create_index("is_active")
        await db.offers.create_index("created_at")
        await db.offers.create_index([("is_active", 1), ("created_at", -1)])
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
