from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",  # Add more origins if necessary
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load('music_genre_predictor_without_encoder.joblib')

# Database URL for SQLite
DATABASE_URL = "sqlite:///./test.db"

# Set up SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# User data model for ML training
class UserData(Base):
    __tablename__ = 'user_data'

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    gender = Column(String)
    genre = Column(String)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for input and output validation
class UserInput(BaseModel):
    age: int
    gender: str

class UserOutput(BaseModel):
    genre: str

# CRUD operation to create a user record in the database
def create_user(db: Session, age: int, gender: str, genre: str):
    db_user = UserData(age=age, gender=gender, genre=genre)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Endpoint 1: predict (Age and Gender -> Favorite Music Genre)
@app.post("/predict", response_model=UserOutput)
def predict(input_data: UserInput, db: Session = Depends(get_db)):
    # Prepare the input features (one-hot encoding for gender)
    gender_input = input_data.gender.lower()
    input_features = pd.DataFrame([[input_data.age, 1 if gender_input == 'male' else 0]], columns=['age', 'gender_male'])
    
    # Predict the genre
    predicted_genre_encoded = model.predict(input_features)[0]

    # Map the predicted genre (0 for Classical, 1 for Rock, etc.)
    genre_mapping = {
        0: "Classical",
        1: "Rock",
        2: "Pop",
        3: "Jazz"
    }
    
    predicted_genre = genre_mapping.get(predicted_genre_encoded, "Unknown")
    
    return {"genre": predicted_genre}

# Endpoint 2: ml (Store data for ML)
@app.post("/ml", response_model=UserOutput)
def store_ml_data(input_data: UserInput, db: Session = Depends(get_db)):
    # Store the user data (age, gender) in the database for future ML training
    db_user = create_user(db, age=input_data.age, gender=input_data.gender, genre="")
    return {"genre": f"Data stored successfully"}
