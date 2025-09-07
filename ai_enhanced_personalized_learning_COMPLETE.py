import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import time
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import math
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Enhanced Personalized Learning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class User:
    user_id: str
    display_name: str
    preferences: Dict[str, Any]
    created_at: datetime
    last_login: datetime

@dataclass
class UserPreferences:
    preferred_topics: List[str]
    preferred_difficulty: str
    preferred_story_length: str
    interests: List[str]
    learning_goals: List[str]

@dataclass
class ReadingSession:
    story_id: str
    story_content: str
    reading_start_time: float
    reading_end_time: float
    reading_duration: float
    word_count: int
    reading_speed_wpm: float

@dataclass
class QuestionResponse:
    question_id: str
    question_text: str
    options: List[str]
    correct_answer: str
    user_answer: str
    is_correct: bool
    response_time: float
    difficulty_level: str
    question_type: str

@dataclass
class QuizSession:
    session_id: str
    story_id: str
    reading_session: ReadingSession
    responses: List[QuestionResponse]
    total_score: int
    total_questions: int
    accuracy_rate: float
    average_response_time: float
    session_date: datetime
    difficulty_level: str

class PersonalizedLearningDB:
    """Enhanced database manager with user management"""

    def __init__(self, db_path: str = "ai_enhanced_learning.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables including user management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Stories table (enhanced with topic categories)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    word_count INTEGER,
                    difficulty_level TEXT,
                    category TEXT,
                    topics TEXT,
                    story_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Reading sessions table (with user_id)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reading_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    story_id INTEGER,
                    reading_start_time REAL,
                    reading_end_time REAL,
                    reading_duration REAL,
                    word_count INTEGER,
                    reading_speed_wpm REAL,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (story_id) REFERENCES stories (id)
                )
            """)

            # Quiz sessions table (with user_id)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    user_id TEXT,
                    story_id INTEGER,
                    reading_session_id INTEGER,
                    total_score INTEGER,
                    total_questions INTEGER,
                    accuracy_rate REAL,
                    average_response_time REAL,
                    difficulty_level TEXT,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (story_id) REFERENCES stories (id),
                    FOREIGN KEY (reading_session_id) REFERENCES reading_sessions (id)
                )
            """)

            # Question responses table  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quiz_session_id TEXT,
                    user_id TEXT,
                    question_id TEXT,
                    question_text TEXT,
                    options TEXT,
                    correct_answer TEXT,
                    user_answer TEXT,
                    is_correct BOOLEAN,
                    response_time REAL,
                    difficulty_level TEXT,
                    question_type TEXT,
                    FOREIGN KEY (quiz_session_id) REFERENCES quiz_sessions (session_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_story_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preferred_topics TEXT,
                    preferred_difficulty TEXT,
                    preferred_story_length TEXT,
                    interests TEXT,
                    learning_goals TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # AI Search queries table (NEW)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_search_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    intent TEXT,
                    results_count INTEGER,
                    query_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            conn.commit()

            # Insert sample users if none exist
            self.create_sample_users()

    def create_sample_users(self):
        """Create sample users for demonstration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if users already exist
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] > 0:
                return

            sample_users = [
                ("STU001", "Alice Johnson"),
                ("STU002", "Bob Smith"), 
                ("STU003", "Carol Davis"),
                ("STU004", "David Wilson"),
                ("STU005", "Emma Brown")
            ]

            for user_id, display_name in sample_users:
                default_prefs = {
                    "preferred_topics": ["adventure", "science"],
                    "preferred_difficulty": "medium",
                    "preferred_story_length": "medium",
                    "interests": ["reading", "learning"],
                    "learning_goals": ["improve comprehension", "increase speed"]
                }

                cursor.execute("""
                    INSERT INTO users (user_id, display_name, preferences)
                    VALUES (?, ?, ?)
                """, (user_id, display_name, json.dumps(default_prefs)))

    def save_ai_search_query(self, user_id: str, query: str, intent: str, results_count: int):
        """Save AI search query for analytics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_search_queries (user_id, query, intent, results_count)
                VALUES (?, ?, ?, ?)
            """, (user_id, query, intent, results_count))
            conn.commit()

    def authenticate_user(self, user_id: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, display_name, preferences, created_at, last_login
                FROM users WHERE user_id = ?
            """, (user_id,))

            result = cursor.fetchone()
            if result:
                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                """, (user_id,))
                conn.commit()

                return {
                    "user_id": result[0],
                    "display_name": result[1],
                    "preferences": json.loads(result[2]) if result[2] else {},
                    "created_at": result[3],
                    "last_login": result[4]
                }
            return None

    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Update user preferences in users table
            cursor.execute("""
                UPDATE users SET preferences = ? WHERE user_id = ?
            """, (json.dumps(preferences), user_id))

            # Insert/Update detailed preferences
            cursor.execute("""
                INSERT OR REPLACE INTO user_story_preferences 
                (user_id, preferred_topics, preferred_difficulty, preferred_story_length, 
                 interests, learning_goals, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                user_id,
                json.dumps(preferences.get("preferred_topics", [])),
                preferences.get("preferred_difficulty", "medium"),
                preferences.get("preferred_story_length", "medium"),
                json.dumps(preferences.get("interests", [])),
                json.dumps(preferences.get("learning_goals", []))
            ))

            conn.commit()

    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT preferences FROM users WHERE user_id = ?
            """, (user_id,))

            result = cursor.fetchone()
            if result and result[0]:
                return json.loads(result[0])

            # Return default preferences that match available options
            return {
                "preferred_topics": ["adventure", "science"],
                "preferred_difficulty": "medium",
                "preferred_story_length": "medium",
                "interests": ["reading"],
                "learning_goals": ["improve comprehension"]
            }

    def save_story(self, title: str, content: str, difficulty_level: str, category: str, topics: List[str] = None, story_type: str = "general") -> int:
        """Save story with enhanced metadata"""
        word_count = len(content.split())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stories (title, content, word_count, difficulty_level, category, topics, story_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (title, content, word_count, difficulty_level, category, 
                  json.dumps(topics) if topics else json.dumps([]), story_type))
            return cursor.lastrowid

    def save_reading_session(self, reading_session: ReadingSession, story_id: int, user_id: str) -> int:
        """Save reading session with user ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reading_sessions 
                (user_id, story_id, reading_start_time, reading_end_time, reading_duration, word_count, reading_speed_wpm)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, story_id, reading_session.reading_start_time, reading_session.reading_end_time,
                  reading_session.reading_duration, reading_session.word_count, reading_session.reading_speed_wpm))
            return cursor.lastrowid

    def save_quiz_session(self, quiz_session: QuizSession, story_id: int, reading_session_id: int, user_id: str):
        """Save complete quiz session with user ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Save quiz session
            cursor.execute("""
                INSERT INTO quiz_sessions 
                (session_id, user_id, story_id, reading_session_id, total_score, total_questions, 
                 accuracy_rate, average_response_time, difficulty_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (quiz_session.session_id, user_id, story_id, reading_session_id, quiz_session.total_score,
                  quiz_session.total_questions, quiz_session.accuracy_rate, quiz_session.average_response_time,
                  quiz_session.difficulty_level))

            # Save individual responses
            for response in quiz_session.responses:
                cursor.execute("""
                    INSERT INTO question_responses 
                    (quiz_session_id, user_id, question_id, question_text, options, correct_answer, 
                     user_answer, is_correct, response_time, difficulty_level, question_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (quiz_session.session_id, user_id, response.question_id, response.question_text,
                      json.dumps(response.options), response.correct_answer, response.user_answer,
                      response.is_correct, response.response_time, response.difficulty_level, response.question_type))

            conn.commit()

    def get_user_performance_data(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get performance data for specific user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    qs.*,
                    rs.reading_duration,
                    rs.reading_speed_wpm,
                    s.title as story_title,
                    s.word_count as story_word_count,
                    s.category as story_category,
                    s.topics as story_topics
                FROM quiz_sessions qs
                JOIN reading_sessions rs ON qs.reading_session_id = rs.id
                JOIN stories s ON qs.story_id = s.id
                WHERE qs.user_id = ?
                ORDER BY qs.session_date DESC
                LIMIT ?
            """, (user_id, limit))

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_all_users(self) -> List[Dict]:
        """Get all registered users"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, display_name, last_login,
                       (SELECT COUNT(*) FROM quiz_sessions WHERE user_id = users.user_id) as total_sessions
                FROM users
                ORDER BY last_login DESC
            """)

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

class EnhancedContentProvider:
    """Enhanced content provider with topic-based story selection"""

    @staticmethod
    def get_available_topics() -> List[str]:
        """Get all available story topics"""
        return [
            "adventure", "science", "mystery", "fantasy", "historical",
            "technology", "nature", "space", "friendship", "family",
            "discovery", "courage", "magic", "robots", "travel"
        ]

    @staticmethod
    def get_story_categories() -> List[str]:
        """Get all available story categories"""
        return ["Fiction", "Science Fiction", "Adventure", "Mystery", "Fantasy", "Educational"]

    @staticmethod
    def get_comprehensive_stories() -> Dict[str, Dict]:
        """Enhanced story collection with topics and categories"""
        return {
            "easy_adventure": {
                "title": "The Friendly Robot's Adventure",
                "content": """
                In a small town called Riverside, there lived a friendly robot named Zap. Unlike other robots 
                who worked in factories, Zap loved to explore and help people in his community. Every morning, 
                he would roll through the town square with his bright blue lights flashing cheerfully.

                One sunny day, Mrs. Garcia was struggling to carry heavy groceries from the market to her house. 
                Her bags were so full that apples and oranges kept rolling out onto the sidewalk. Zap saw her 
                trouble and quickly rolled over to help.

                "Don't worry, Mrs. Garcia!" beeped Zap in his cheerful electronic voice. "I can carry all your 
                groceries safely!" With his strong mechanical arms, he carefully gathered all the scattered fruit 
                and organized the bags perfectly.

                As they walked to Mrs. Garcia's house, other neighbors saw Zap helping. Soon, children were 
                following them, asking Zap questions about how he worked. Zap was happy to explain, showing them 
                his different parts and how they helped him be useful.

                From that day forward, Zap became the most popular helper in Riverside. He learned that the best 
                way to make friends was to be kind and helpful to everyone he met.
                """,
                "category": "Adventure",
                "topics": ["robots", "friendship", "community", "kindness"],
                "difficulty": "easy",
                "questions": [
                    {
                        "text": "Where did Zap live?",
                        "options": ["Big City", "Riverside", "Factory Town", "Robot Village"],
                        "correct": "Riverside",
                        "type": "detail_comprehension"
                    },
                    {
                        "text": "What made Zap different from other robots?",
                        "options": ["He was blue", "He loved to explore and help people", "He was very fast", "He could talk"],
                        "correct": "He loved to explore and help people",
                        "type": "character_analysis"
                    },
                    {
                        "text": "What lesson did Zap learn?",
                        "options": ["Speed is important", "Being kind makes friends", "Work is better than play", "Robots are superior"],
                        "correct": "Being kind makes friends",
                        "type": "theme_understanding"
                    }
                ]
            },
            "medium_science": {
                "title": "The Discovery of Quantum Crystals",
                "content": """
                Dr. Sarah Chen stood in her laboratory, examining the strange crystalline formation that had appeared 
                overnight in her quantum physics experiment. The crystal seemed to emit a soft blue glow and appeared 
                to be vibrating at frequencies that defied conventional understanding of matter.

                For months, Sarah had been working on a project to create stable quantum states in solid matter. 
                Her research aimed to develop materials that could revolutionize computing and energy storage. But this 
                crystal was beyond anything she had theorized.

                As she carefully approached the crystal with her measurement devices, something extraordinary happened. 
                The crystal's glow intensified, and the laboratory equipment began displaying readings that seemed 
                impossible. The crystal appeared to be existing in multiple quantum states simultaneously, something 
                that should only occur at the subatomic level.

                Sarah realized she had stumbled upon a breakthrough that could change humanity's understanding of 
                physics. The crystal seemed to bridge the gap between quantum mechanics and macroscopic reality. 
                However, she also understood the enormous responsibility that came with such a discovery.

                She decided to proceed carefully, documenting every observation while considering the ethical implications 
                of her find. The crystal represented both incredible potential and significant danger if misused. Sarah 
                knew that proper scientific protocols and international cooperation would be essential for safely 
                exploring this new frontier.
                """,
                "category": "Science Fiction",
                "topics": ["science", "discovery", "technology", "responsibility"],
                "difficulty": "medium",
                "questions": [
                    {
                        "text": "What was unique about the crystal Sarah discovered?",
                        "options": ["It was very large", "It existed in multiple quantum states", "It was extremely hot", "It could move by itself"],
                        "correct": "It existed in multiple quantum states",
                        "type": "plot_detail"
                    },
                    {
                        "text": "What was Sarah's main concern about her discovery?",
                        "options": ["Lack of funding", "The responsibility and potential danger", "Patent issues", "Laboratory safety"],
                        "correct": "The responsibility and potential danger",
                        "type": "character_motivation"
                    },
                    {
                        "text": "What does Sarah's decision to proceed carefully show about her character?",
                        "options": ["She is fearful", "She is ethical and responsible", "She is indecisive", "She lacks confidence"],
                        "correct": "She is ethical and responsible",
                        "type": "character_analysis"
                    },
                    {
                        "text": "What field of study was Sarah working in?",
                        "options": ["Biology", "Chemistry", "Quantum Physics", "Engineering"],
                        "correct": "Quantum Physics",
                        "type": "detail_comprehension"
                    }
                ]
            },
            "hard_mystery": {
                "title": "The Cipher of the Lost Mathematician",
                "content": """
                Professor Elena Vasquez had spent three decades studying the works of historical mathematicians, but 
                nothing had prepared her for the encrypted journal she found hidden in the university's archive basement. 
                The journal belonged to Dr. Magnus Euler, a brilliant mathematician who had mysteriously disappeared 
                in 1847, leaving behind only cryptic mathematical formulations.

                The journal was filled with complex equations that seemed to predict natural phenomena with unprecedented 
                accuracy. Elena discovered that Euler had apparently developed a mathematical framework that could model 
                weather patterns, seismic activity, and even market fluctuations decades before such analysis was 
                thought possible.

                As Elena worked to decode the cipher, she realized that Euler hadn't just created a predictive model – 
                he had discovered fundamental mathematical relationships that governed seemingly random events. The 
                implications were staggering: if his work was valid, it suggested that chaos theory and deterministic 
                mathematics were more deeply connected than anyone had imagined.

                However, the deeper Elena delved into the journal, the more she understood why Euler might have 
                disappeared. His final entries spoke of powerful individuals who sought to exploit his discoveries 
                for personal gain, and of his growing fear about the consequences of his work falling into the wrong hands.

                Elena faced a profound dilemma: should she publish Euler's groundbreaking work and revolutionize 
                mathematics and science, or should she protect it from potential misuse? The weight of scientific 
                responsibility pressed heavily upon her as she contemplated the far-reaching implications of her discovery.

                In her final analysis, Elena decided to release the work gradually, establishing an international 
                committee of mathematicians and ethicists to oversee its application. She believed that scientific 
                knowledge belonged to humanity, but must be stewarded responsibly.
                """,
                "category": "Mystery",
                "topics": ["mystery", "mathematics", "history", "ethics", "discovery"],
                "difficulty": "hard",
                "questions": [
                    {
                        "text": "What made Dr. Magnus Euler's journal particularly significant?",
                        "options": ["It was very old", "It contained predictive mathematical models", "It was written in code", "It was valuable"],
                        "correct": "It contained predictive mathematical models",
                        "type": "concept_understanding"
                    },
                    {
                        "text": "Why might Euler have disappeared according to the story?",
                        "options": ["He was ill", "He feared his work would be misused", "He was offered a new job", "He was persecuted"],
                        "correct": "He feared his work would be misused",
                        "type": "inference"
                    },
                    {
                        "text": "What dilemma did Elena face?",
                        "options": ["Funding issues", "Whether to publish or protect the work", "Language barriers", "Time constraints"],
                        "correct": "Whether to publish or protect the work",
                        "type": "conflict_analysis"
                    },
                    {
                        "text": "How did Elena ultimately decide to handle the discovery?",
                        "options": ["Keep it secret", "Sell it", "Release it gradually with oversight", "Destroy it"],
                        "correct": "Release it gradually with oversight",
                        "type": "plot_resolution"
                    },
                    {
                        "text": "What does Elena's final decision reveal about her values?",
                        "options": ["She prioritizes fame", "She values responsible stewardship of knowledge", "She is risk-averse", "She trusts institutions blindly"],
                        "correct": "She values responsible stewardship of knowledge",
                        "type": "character_analysis"
                    }
                ]
            },
            "medium_fantasy": {
                "title": "The Guardian of the Crystal Forest",
                "content": """
                In the mystical realm of Aethermoor, young Maya inherited an extraordinary responsibility from her 
                grandmother: becoming the Guardian of the Crystal Forest. This ancient woodland was home to sentient 
                crystal trees that maintained the magical balance of their world, but a dark corruption was spreading 
                through the forest, turning the beautiful crystals black and lifeless.

                Maya discovered that the corruption originated from a powerful artifact called the Shadow Heart, which 
                had been awakened by a group of treasure hunters who didn't understand its dangerous nature. The artifact 
                fed on life energy, growing stronger as it consumed the forest's magic.

                To save the Crystal Forest, Maya had to master three ancient guardian abilities: the power to communicate 
                with the crystal trees, the skill to purify corrupted magic, and the wisdom to restore natural balance. 
                Each ability required not just magical talent, but deep understanding of the interconnected nature of 
                all living things.

                During her quest, Maya learned that being a guardian meant more than having power – it required sacrifice, 
                patience, and the courage to make difficult decisions for the greater good. She had to convince the 
                treasure hunters to help her contain the Shadow Heart, even though it meant giving up their chance for wealth.

                In the final confrontation, Maya realized that the Shadow Heart couldn't be destroyed – it could only 
                be balanced with an equal force of light. She made the ultimate guardian's choice, binding part of her 
                own life essence to create a permanent seal that would protect the forest for generations to come.
                """,
                "category": "Fantasy",
                "topics": ["fantasy", "nature", "magic", "responsibility", "courage"],
                "difficulty": "medium",
                "questions": [
                    {
                        "text": "What was Maya's inherited responsibility?",
                        "options": ["Ruling a kingdom", "Becoming the Guardian of the Crystal Forest", "Leading an army", "Managing a library"],
                        "correct": "Becoming the Guardian of the Crystal Forest",
                        "type": "plot_detail"
                    },
                    {
                        "text": "What caused the corruption in the forest?",
                        "options": ["Natural decay", "The Shadow Heart artifact", "Climate change", "Enemy invasion"],
                        "correct": "The Shadow Heart artifact",
                        "type": "cause_effect"
                    },
                    {
                        "text": "What three abilities did Maya need to master?",
                        "options": ["Fighting, flying, hiding", "Communication, purification, balance", "Reading, writing, calculating", "Singing, dancing, painting"],
                        "correct": "Communication, purification, balance",
                        "type": "detail_comprehension"
                    },
                    {
                        "text": "How did Maya ultimately deal with the Shadow Heart?",
                        "options": ["Destroyed it", "Created a balance seal with her life essence", "Buried it", "Gave it away"],
                        "correct": "Created a balance seal with her life essence",
                        "type": "plot_resolution"
                    }
                ]
            }
        }

    @classmethod
    def get_stories_by_preferences(cls, preferences: Dict) -> Dict[str, Dict]:
        """Filter stories based on user preferences"""
        all_stories = cls.get_comprehensive_stories()
        preferred_topics = set(preferences.get("preferred_topics", []))
        preferred_difficulty = preferences.get("preferred_difficulty", "medium")

        # If no specific preferences, return all stories
        if not preferred_topics:
            return all_stories

        filtered_stories = {}

        for story_key, story_data in all_stories.items():
            story_topics = set(story_data.get("topics", []))
            story_difficulty = story_data.get("difficulty", "medium")

            # Check if story matches user's topic preferences
            if story_topics.intersection(preferred_topics):
                filtered_stories[story_key] = story_data
            # Also include stories of preferred difficulty
            elif story_difficulty == preferred_difficulty:
                filtered_stories[story_key] = story_data

        # If no matches found, return a few default stories
        if not filtered_stories:
            # Return first few stories as fallback
            story_keys = list(all_stories.keys())[:3]
            filtered_stories = {key: all_stories[key] for key in story_keys}

        return filtered_stories

class LearningAnalyzer:
    """Enhanced analyzer with user-specific insights"""

    @staticmethod
    def calculate_reading_metrics(sessions: List[Dict]) -> Dict[str, float]:
        """Calculate reading performance metrics"""
        if not sessions:
            return {}

        reading_speeds = [s["reading_speed_wpm"] for s in sessions if s["reading_speed_wpm"]]
        reading_durations = [s["reading_duration"] for s in sessions if s["reading_duration"]]

        return {
            "avg_reading_speed": np.mean(reading_speeds) if reading_speeds else 0,
            "reading_speed_trend": np.polyfit(range(len(reading_speeds)), reading_speeds, 1)[0] if len(reading_speeds) > 1 else 0,
            "avg_reading_duration": np.mean(reading_durations) if reading_durations else 0,
            "reading_consistency": 1 - (np.std(reading_speeds) / np.mean(reading_speeds)) if reading_speeds and np.mean(reading_speeds) > 0 else 0
        }

    @staticmethod
    def calculate_quiz_metrics(sessions: List[Dict]) -> Dict[str, float]:
        """Calculate quiz performance metrics"""
        if not sessions:
            return {}

        accuracies = [s["accuracy_rate"] for s in sessions]
        response_times = [s["average_response_time"] for s in sessions]

        return {
            "avg_accuracy": np.mean(accuracies) if accuracies else 0,
            "accuracy_trend": np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0,
            "avg_response_time": np.mean(response_times) if response_times else 0,
            "response_time_trend": np.polyfit(range(len(response_times)), response_times, 1)[0] if len(response_times) > 1 else 0,
            "consistency_score": 1 - (np.std(accuracies) / np.mean(accuracies)) if accuracies and np.mean(accuracies) > 0 else 0
        }

    @staticmethod
    def generate_personalized_insights(reading_metrics: Dict, quiz_metrics: Dict, recent_sessions: List[Dict], user_preferences: Dict) -> List[str]:
        """Generate personalized learning insights based on user data"""
        insights = []

        # Reading speed analysis
        avg_speed = reading_metrics.get("avg_reading_speed", 0)
        if avg_speed < 150:
            insights.append(f"📚 Your reading speed ({avg_speed:.1f} WPM) can be improved. Try practicing with your favorite topics: {', '.join(user_preferences.get('preferred_topics', ['adventure']))}")
        elif avg_speed > 250:
            insights.append(f"🚀 Excellent reading speed ({avg_speed:.1f} WPM)! You're reading faster than average.")

        if reading_metrics.get("reading_speed_trend", 0) > 5:
            insights.append("📈 Great progress! Your reading speed is improving over time.")
        elif reading_metrics.get("reading_speed_trend", 0) < -5:
            insights.append("📉 Your reading speed has been declining. Consider taking breaks between sessions.")

        # Accuracy analysis with personalization
        avg_accuracy = quiz_metrics.get("avg_accuracy", 0)
        if avg_accuracy > 0.8:
            preferred_difficulty = user_preferences.get("preferred_difficulty", "medium")
            next_level = {"easy": "medium", "medium": "hard", "hard": "expert"}
            if preferred_difficulty in next_level:
                insights.append(f"🎯 Excellent comprehension ({avg_accuracy:.1%})! Consider trying {next_level[preferred_difficulty]} level stories.")
        elif avg_accuracy < 0.6:
            insights.append("💡 Focus on careful reading. Try stories about your interests to improve engagement.")

        return insights if insights else ["Keep practicing to see more personalized insights!"]

    @staticmethod
    def recommend_difficulty_level(recent_performance: List[Dict]) -> str:
        """Recommend appropriate difficulty level based on performance"""
        if not recent_performance:
            return "medium"

        recent_accuracy = np.mean([s["accuracy_rate"] for s in recent_performance[-5:]])
        recent_speed = np.mean([s["reading_speed_wpm"] for s in recent_performance[-5:]])

        if recent_accuracy > 0.85 and recent_speed > 200:
            return "hard"
        elif recent_accuracy > 0.7 and recent_speed > 150:
            return "medium"
        else:
            return "easy"

class AISearchEngine:
    """Advanced AI-powered search engine for personalized learning"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.story_vectors = None
        self.story_texts = []
        self.story_metadata = []
        self.knowledge_base = self._build_knowledge_base()
        self.initialize_search_index()

    def _build_knowledge_base(self) -> Dict:
        """Build AI knowledge base for intelligent responses"""
        return {
            "reading_strategies": {
                "speed": [
                    "Practice skimming and scanning techniques",
                    "Reduce subvocalization (inner voice)",
                    "Use a pointer or finger to guide reading",
                    "Read in chunks rather than word by word",
                    "Practice with familiar topics first"
                ],
                "comprehension": [
                    "Preview the text before reading",
                    "Ask questions while reading",
                    "Summarize paragraphs in your own words",
                    "Connect new information to prior knowledge",
                    "Visualize what you're reading"
                ],
                "vocabulary": [
                    "Use context clues to understand new words",
                    "Keep a vocabulary journal",
                    "Practice with word roots and prefixes",
                    "Use new words in your own sentences",
                    "Read diverse topics to encounter varied vocabulary"
                ]
            },
            "learning_concepts": {
                "plot": "The sequence of events in a story, including exposition, rising action, climax, falling action, and resolution",
                "character": "People, animals, or entities in a story who drive the narrative forward",
                "theme": "The central message or underlying meaning of a story",
                "setting": "The time and place where a story occurs",
                "inference": "Drawing conclusions based on evidence and reasoning rather than explicit statements",
                "main_idea": "The central point or primary message of a text or passage",
                "supporting_details": "Facts, examples, or evidence that support the main idea",
                "cause_effect": "The relationship between events where one event leads to another"
            },
            "study_tips": {
                "focus": [
                    "Find a quiet, well-lit study space",
                    "Remove distractions like phones and notifications",
                    "Take regular breaks using the Pomodoro Technique",
                    "Set specific goals for each study session",
                    "Practice mindfulness to improve concentration"
                ],
                "motivation": [
                    "Set small, achievable goals",
                    "Track your progress visually",
                    "Reward yourself for milestones",
                    "Find a study buddy or accountability partner",
                    "Remember your long-term learning objectives"
                ],
                "memory": [
                    "Use spaced repetition for better retention",
                    "Create mind maps or visual organizers",
                    "Teach concepts to someone else",
                    "Make connections between new and existing knowledge",
                    "Use mnemonics and memory palace techniques"
                ]
            }
        }

    def initialize_search_index(self):
        """Initialize search index with story content"""
        stories = EnhancedContentProvider.get_comprehensive_stories()

        self.story_texts = []
        self.story_metadata = []

        for key, story in stories.items():
            # Combine title, content, and topics for comprehensive search
            searchable_text = f"{story['title']} {story['content']} {' '.join(story['topics'])}"
            self.story_texts.append(searchable_text)
            self.story_metadata.append({
                'key': key,
                'title': story['title'],
                'topics': story['topics'],
                'difficulty': story['difficulty'],
                'category': story['category']
            })

        if self.story_texts:
            self.story_vectors = self.vectorizer.fit_transform(self.story_texts)

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform semantic search across story content"""
        if not self.story_vectors:
            return []

        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.story_vectors).flatten()

            # Get top-k most similar stories
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        'metadata': self.story_metadata[idx],
                        'similarity': similarities[idx],
                        'snippet': self._extract_snippet(self.story_texts[idx], query)
                    })

            return results
        except:
            return []

    def _extract_snippet(self, text: str, query: str, snippet_length: int = 200) -> str:
        """Extract relevant snippet from text based on query"""
        query_words = query.lower().split()
        text_lower = text.lower()

        # Find the best position that contains query words
        best_pos = 0
        best_score = 0

        words = text.split()
        for i in range(len(words) - 20):
            chunk = ' '.join(words[i:i+30]).lower()
            score = sum(1 for word in query_words if word in chunk)
            if score > best_score:
                best_score = score
                best_pos = i

        # Extract snippet around the best position
        start_pos = max(0, best_pos - 10)
        end_pos = min(len(words), best_pos + 40)
        snippet = ' '.join(words[start_pos:end_pos])

        return snippet[:snippet_length] + "..." if len(snippet) > snippet_length else snippet

    def ai_assistant_response(self, query: str, user_context: Dict = None) -> str:
        """Generate AI assistant response based on query and context"""
        query_lower = query.lower()

        # Context-aware responses based on user data
        if user_context:
            user_prefs = user_context.get('preferences', {})
            performance = user_context.get('performance', [])

            # Calculate performance metrics for context
            if performance:
                avg_speed = np.mean([p.get('reading_speed_wpm', 0) for p in performance])
                avg_accuracy = np.mean([p.get('accuracy_rate', 0) for p in performance])

                # Performance-based responses
                if any(word in query_lower for word in ['slow', 'speed', 'faster', 'quick']):
                    if avg_speed < 150:
                        return f"📈 Your current reading speed is {avg_speed:.0f} WPM. Here are personalized techniques to improve: " +                                " • ".join(random.sample(self.knowledge_base['reading_strategies']['speed'], 3))
                    else:
                        return f"🚀 Great job on your reading speed ({avg_speed:.0f} WPM)! Here are ways to maintain and optimize it: " +                                " • ".join(random.sample(self.knowledge_base['reading_strategies']['speed'], 3))

                if any(word in query_lower for word in ['understand', 'comprehension', 'confused', 'difficult']):
                    return f"🧠 Comprehension Support (Current accuracy: {avg_accuracy:.1%}): " +                            "Key strategies: " + " • ".join(random.sample(self.knowledge_base['reading_strategies']['comprehension'], 3))

                if any(word in query_lower for word in ['vocabulary', 'words', 'meaning', 'definition']):
                    return "📚 Here are effective vocabulary building strategies: " +                            " • ".join(random.sample(self.knowledge_base['reading_strategies']['vocabulary'], 3))

        # General topic-based responses
        if any(word in query_lower for word in ['plot', 'story', 'narrative']):
            return f"📖 Plot Analysis: {self.knowledge_base['learning_concepts']['plot']}. "                    "When reading stories, look for the five key elements: exposition (setup), "                    "rising action (building tension), climax (turning point), falling action (resolution begins), "                    "and resolution (conclusion)."

        if any(word in query_lower for word in ['character', 'protagonist', 'hero']):
            return f"👤 Character Analysis: {self.knowledge_base['learning_concepts']['character']}. "                    "Pay attention to character motivations, development throughout the story, "                    "and how they interact with other characters and the plot."

        if any(word in query_lower for word in ['theme', 'message', 'meaning']):
            return f"💡 Theme Understanding: {self.knowledge_base['learning_concepts']['theme']}. "                    "Look for repeated ideas, symbols, and the author's perspective on life or human nature."

        if any(word in query_lower for word in ['focus', 'concentration', 'distracted']):
            return "🎯 Focus Improvement Tips: " + " • ".join(self.knowledge_base['study_tips']['focus'])

        if any(word in query_lower for word in ['motivation', 'bored', 'unmotivated']):
            return "💪 Motivation Strategies: " + " • ".join(self.knowledge_base['study_tips']['motivation'])

        if any(word in query_lower for word in ['remember', 'memory', 'forget', 'retention']):
            return "🧠 Memory Enhancement: " + " • ".join(self.knowledge_base['study_tips']['memory'])

        # Default intelligent response
        return self._generate_contextual_response(query, user_context)

    def _generate_contextual_response(self, query: str, user_context: Dict) -> str:
        """Generate contextual AI response"""
        responses = [
            "I'm here to help you improve your reading and learning! Could you be more specific about what you'd like to know?",
            f"Based on your question about '{query}', I'd recommend exploring our reading strategies or asking about specific learning concepts.",
            "Great question! I can help with reading comprehension, speed improvement, vocabulary building, or study techniques. What interests you most?",
            f"For '{query}', I can provide personalized advice based on your reading performance and preferences. What specific aspect would you like to focus on?"
        ]

        return random.choice(responses) + " Try asking about topics like 'how to read faster', 'improve comprehension', or 'study tips'."

    def smart_recommendations(self, user_context: Dict) -> List[Dict]:
        """Generate AI-powered learning recommendations"""
        recommendations = []

        if not user_context:
            return recommendations

        preferences = user_context.get('preferences', {})
        performance = user_context.get('performance', [])

        # Analyze user's learning patterns
        if performance:
            avg_accuracy = np.mean([p.get('accuracy_rate', 0) for p in performance])
            avg_speed = np.mean([p.get('reading_speed_wpm', 0) for p in performance])

            # Speed-based recommendations
            if avg_speed < 150:
                recommendations.append({
                    'type': 'skill_improvement',
                    'title': 'Boost Your Reading Speed',
                    'description': 'Personalized speed reading techniques for your current level',
                    'action': 'Practice with familiar topics to build fluency',
                    'priority': 'high'
                })

            # Accuracy-based recommendations
            if avg_accuracy < 0.75:
                recommendations.append({
                    'type': 'comprehension',
                    'title': 'Strengthen Reading Comprehension',
                    'description': 'Targeted exercises to improve understanding',
                    'action': 'Focus on preview and summary techniques',
                    'priority': 'high'
                })
            elif avg_accuracy > 0.9:
                recommendations.append({
                    'type': 'challenge',
                    'title': 'Ready for Advanced Content',
                    'description': 'You\'re excelling! Time for harder materials',
                    'action': 'Try complex stories or new topic areas',
                    'priority': 'medium'
                })

        # Interest-based recommendations
        preferred_topics = preferences.get('preferred_topics', [])
        if preferred_topics:
            recommendations.append({
                'type': 'content',
                'title': f'Explore More {preferred_topics[0].title()} Stories',
                'description': 'Discover new stories matching your interests',
                'action': f'Search for advanced {preferred_topics[0]} content',
                'priority': 'medium'
            })

        # Goal-based recommendations
        learning_goals = preferences.get('learning_goals', [])
        for goal in learning_goals[:2]:  # Top 2 goals
            if goal == 'increase speed':
                recommendations.append({
                    'type': 'goal_oriented',
                    'title': 'Speed Reading Practice Plan',
                    'description': 'Structured exercises to achieve your speed goals',
                    'action': 'Follow daily 15-minute speed drills',
                    'priority': 'high'
                })
            elif goal == 'improve comprehension':
                recommendations.append({
                    'type': 'goal_oriented',
                    'title': 'Comprehension Mastery Program',
                    'description': 'Step-by-step comprehension improvement',
                    'action': 'Practice active reading strategies',
                    'priority': 'high'
                })

        return recommendations[:4]  # Return top 4 recommendations

    def natural_language_query(self, query: str, user_context: Dict = None) -> Dict:
        """Process natural language queries and return structured results"""
        query_lower = query.lower()

        # Detect query intent
        intent = self._detect_query_intent(query_lower)

        result = {
            'intent': intent,
            'response': '',
            'suggestions': [],
            'search_results': [],
            'recommendations': []
        }

        if intent == 'search':
            # Perform semantic search
            search_results = self.semantic_search(query, top_k=3)
            result['search_results'] = search_results
            result['response'] = f"Found {len(search_results)} stories related to '{query}'"

        elif intent == 'help':
            # Generate AI assistant response
            result['response'] = self.ai_assistant_response(query, user_context)
            result['suggestions'] = [
                "How can I read faster?",
                "Explain the main theme",
                "What are good study techniques?",
                "How to improve comprehension?"
            ]

        elif intent == 'recommend':
            # Generate smart recommendations
            recommendations = self.smart_recommendations(user_context)
            result['recommendations'] = recommendations
            result['response'] = f"Here are {len(recommendations)} personalized recommendations for you!"

        else:
            # Default response with search
            search_results = self.semantic_search(query, top_k=2)
            ai_response = self.ai_assistant_response(query, user_context)

            result['search_results'] = search_results
            result['response'] = ai_response
            result['suggestions'] = [
                f"Search for '{query}' stories",
                "Get help with reading strategies",
                "Find personalized recommendations"
            ]

        return result

    def _detect_query_intent(self, query: str) -> str:
        """Detect the intent of user's natural language query"""
        search_keywords = ['find', 'search', 'look for', 'show me', 'stories about']
        help_keywords = ['how', 'why', 'what', 'explain', 'help', 'improve', 'better', 'tips']
        recommend_keywords = ['recommend', 'suggest', 'advice', 'best', 'should i', 'what next']

        if any(keyword in query for keyword in search_keywords):
            return 'search'
        elif any(keyword in query for keyword in recommend_keywords):
            return 'recommend'
        elif any(keyword in query for keyword in help_keywords):
            return 'help'
        else:
            return 'general'

# [Include all the progress insights functions from the previous implementation]
def progress_insights_page():
    """Comprehensive progress insights and recommendations"""
    st.header(f"🎯 Progress Insights & Recommendations - {st.session_state.current_user['display_name']}")

    # Get user data
    performance_data = st.session_state.db.get_user_performance_data(
        st.session_state.current_user["user_id"]
    )
    user_preferences = st.session_state.db.get_user_preferences(st.session_state.current_user["user_id"])

    if not performance_data:
        st.info("📈 Complete some reading sessions to unlock detailed progress insights!")

        # Show what insights will be available
        st.subheader("🔮 Coming Soon - Your Personal Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 Performance Analytics:**
            - Learning curve progression
            - Reading speed trends
            - Accuracy improvement patterns
            - Response time optimization
            """)

        with col2:
            st.markdown("""
            **🎯 Personalized Recommendations:**
            - Optimal study times
            - Topic-specific strategies
            - Difficulty progression plans
            - Goal achievement roadmaps
            """)

        return

    # Calculate comprehensive metrics
    reading_metrics = st.session_state.analyzer.calculate_reading_metrics(performance_data)
    quiz_metrics = st.session_state.analyzer.calculate_quiz_metrics(performance_data)

    # Create insights sections
    create_learning_curve_analysis(performance_data)
    create_time_based_analysis(performance_data)
    create_topic_mastery_analysis(performance_data, user_preferences)
    create_personalized_recommendations(reading_metrics, quiz_metrics, performance_data, user_preferences)
    create_goal_progress_tracking(performance_data, user_preferences)
    create_future_predictions(performance_data)

def create_learning_curve_analysis(performance_data: List[Dict]):
    """Create learning curve analysis"""
    st.subheader("📈 Learning Curve Analysis")

    df = pd.DataFrame(performance_data)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.sort_values("session_date")
    df["session_number"] = range(1, len(df) + 1)

    # Calculate moving averages
    df["accuracy_ma"] = df["accuracy_rate"].rolling(window=3, min_periods=1).mean()
    df["speed_ma"] = df["reading_speed_wpm"].rolling(window=3, min_periods=1).mean()

    # Create learning curve visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy Learning Curve", "Reading Speed Progress", 
                       "Response Time Improvement", "Consistency Score"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Accuracy curve
    fig.add_trace(
        go.Scatter(
            x=df["session_number"],
            y=df["accuracy_rate"] * 100,
            mode="markers",
            name="Actual Accuracy",
            opacity=0.6,
            marker=dict(color="lightblue")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["session_number"],
            y=df["accuracy_ma"] * 100,
            mode="lines",
            name="Accuracy Trend",
            line=dict(color="blue", width=3)
        ),
        row=1, col=1
    )

    # Reading speed curve
    fig.add_trace(
        go.Scatter(
            x=df["session_number"],
            y=df["reading_speed_wpm"],
            mode="markers",
            name="Actual Speed",
            opacity=0.6,
            marker=dict(color="lightgreen")
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df["session_number"],
            y=df["speed_ma"],
            mode="lines",
            name="Speed Trend",
            line=dict(color="green", width=3)
        ),
        row=1, col=2
    )

    # Response time improvement
    fig.add_trace(
        go.Scatter(
            x=df["session_number"],
            y=df["average_response_time"],
            mode="lines+markers",
            name="Response Time",
            line=dict(color="orange")
        ),
        row=2, col=1
    )

    # Consistency score (inverse of coefficient of variation)
    if len(df) > 3:
        consistency_scores = []
        for i in range(3, len(df) + 1):
            recent_accuracy = df["accuracy_rate"].iloc[max(0, i-3):i]
            if len(recent_accuracy) > 1 and recent_accuracy.mean() > 0:
                cv = recent_accuracy.std() / recent_accuracy.mean()
                consistency = 1 - min(cv, 1)  # Cap at 1
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(0.5)  # Default middle value

        fig.add_trace(
            go.Scatter(
                x=list(range(4, len(df) + 1)),
                y=consistency_scores,
                mode="lines+markers",
                name="Consistency",
                line=dict(color="purple")
            ),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=True, title_text="Learning Progress Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate improvement metrics
    if len(df) > 1:
        accuracy_improvement = (df["accuracy_rate"].iloc[-1] - df["accuracy_rate"].iloc[0]) * 100
        speed_improvement = df["reading_speed_wpm"].iloc[-1] - df["reading_speed_wpm"].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy Improvement", f"{accuracy_improvement:+.1f}%")
        with col2:
            st.metric("Speed Improvement", f"{speed_improvement:+.1f} WPM")
        with col3:
            sessions_completed = len(df)
            st.metric("Sessions Completed", sessions_completed)

def create_time_based_analysis(performance_data: List[Dict]):
    """Analyze performance patterns by time"""
    st.subheader("⏰ Time-Based Learning Patterns")

    df = pd.DataFrame(performance_data)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df["hour"] = df["session_date"].dt.hour
    df["day_of_week"] = df["session_date"].dt.day_name()

    col1, col2 = st.columns(2)

    with col1:
        # Performance by hour of day
        if len(df["hour"].unique()) > 1:
            hourly_performance = df.groupby("hour").agg({
                "accuracy_rate": "mean",
                "reading_speed_wpm": "mean",
                "average_response_time": "mean"
            }).reset_index()

            fig = px.line(
                hourly_performance,
                x="hour",
                y="accuracy_rate",
                title="Performance by Hour of Day",
                markers=True
            )
            fig.update_xaxes(title="Hour of Day")
            fig.update_yaxes(title="Average Accuracy")
            st.plotly_chart(fig, use_container_width=True)

            # Find optimal time
            best_hour = hourly_performance.loc[hourly_performance["accuracy_rate"].idxmax(), "hour"]
            st.info(f"🎯 Your optimal learning time: {best_hour}:00 - {best_hour+1}:00")
        else:
            st.info("Complete more sessions at different times to see hourly patterns!")

    with col2:
        # Performance by day of week
        if len(df["day_of_week"].unique()) > 1:
            daily_performance = df.groupby("day_of_week").agg({
                "accuracy_rate": "mean",
                "reading_speed_wpm": "mean"
            }).reset_index()

            # Sort by day order
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_performance["day_of_week"] = pd.Categorical(daily_performance["day_of_week"], categories=day_order, ordered=True)
            daily_performance = daily_performance.sort_values("day_of_week")

            fig = px.bar(
                daily_performance,
                x="day_of_week",
                y="accuracy_rate",
                title="Performance by Day of Week",
                color="accuracy_rate",
                color_continuous_scale="viridis"
            )
            fig.update_xaxes(title="Day of Week")
            fig.update_yaxes(title="Average Accuracy")
            st.plotly_chart(fig, use_container_width=True)

            # Find best day
            best_day = daily_performance.loc[daily_performance["accuracy_rate"].idxmax(), "day_of_week"]
            st.info(f"📅 Your best learning day: {best_day}")
        else:
            st.info("Complete sessions on different days to see daily patterns!")

def create_topic_mastery_analysis(performance_data: List[Dict], user_preferences: Dict):
    """Analyze performance by topic and identify mastery levels"""
    st.subheader("📚 Topic Mastery Analysis")

    # Analyze performance by story topics
    topic_performance = {}
    preferred_topics = set(user_preferences.get("preferred_topics", []))

    for session in performance_data:
        story_topics = json.loads(session.get("story_topics", "[]"))
        for topic in story_topics:
            if topic not in topic_performance:
                topic_performance[topic] = {
                    "accuracies": [],
                    "speeds": [],
                    "times": [],
                    "sessions": 0
                }
            topic_performance[topic]["accuracies"].append(session["accuracy_rate"])
            topic_performance[topic]["speeds"].append(session["reading_speed_wpm"])
            topic_performance[topic]["times"].append(session["average_response_time"])
            topic_performance[topic]["sessions"] += 1

    if topic_performance:
        # Create topic mastery dataframe
        topic_data = []
        for topic, data in topic_performance.items():
            avg_accuracy = np.mean(data["accuracies"])
            avg_speed = np.mean(data["speeds"])
            sessions = data["sessions"]
            is_preferred = topic in preferred_topics

            # Determine mastery level
            if avg_accuracy >= 0.9 and sessions >= 3:
                mastery = "Expert"
            elif avg_accuracy >= 0.8 and sessions >= 2:
                mastery = "Proficient"
            elif avg_accuracy >= 0.7:
                mastery = "Developing"
            else:
                mastery = "Novice"

            topic_data.append({
                "topic": topic.title(),
                "accuracy": avg_accuracy,
                "speed": avg_speed,
                "sessions": sessions,
                "mastery": mastery,
                "preferred": is_preferred
            })

        topic_df = pd.DataFrame(topic_data)

        # Visualize topic mastery
        fig = px.scatter(
            topic_df,
            x="speed",
            y="accuracy",
            size="sessions",
            color="mastery",
            hover_data=["topic"],
            title="Topic Mastery Map",
            labels={"speed": "Reading Speed (WPM)", "accuracy": "Accuracy Rate"},
            color_discrete_map={
                "Expert": "green",
                "Proficient": "blue", 
                "Developing": "orange",
                "Novice": "red"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        # Topic recommendations
        st.subheader("🎯 Topic-Based Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🌟 Your Strongest Topics:**")
            expert_topics = topic_df[topic_df["mastery"] == "Expert"]["topic"].tolist()
            proficient_topics = topic_df[topic_df["mastery"] == "Proficient"]["topic"].tolist()

            strong_topics = expert_topics + proficient_topics
            if strong_topics:
                for topic in strong_topics[:3]:
                    mastery = topic_df[topic_df["topic"] == topic]["mastery"].iloc[0]
                    accuracy = topic_df[topic_df["topic"] == topic]["accuracy"].iloc[0]
                    st.write(f"• {topic} ({mastery} - {accuracy:.1%})")
            else:
                st.write("Keep practicing to identify your strongest topics!")

        with col2:
            st.markdown("**💪 Areas for Improvement:**")
            developing_topics = topic_df[topic_df["mastery"].isin(["Developing", "Novice"])]["topic"].tolist()

            if developing_topics:
                for topic in developing_topics[:3]:
                    mastery = topic_df[topic_df["topic"] == topic]["mastery"].iloc[0]
                    accuracy = topic_df[topic_df["topic"] == topic]["accuracy"].iloc[0]
                    st.write(f"• {topic} ({mastery} - {accuracy:.1%})")
            else:
                st.write("Great job! You're performing well across all topics.")
    else:
        st.info("Complete more sessions to see topic-specific insights!")

def create_personalized_recommendations(reading_metrics: Dict, quiz_metrics: Dict, 
                                      performance_data: List[Dict], user_preferences: Dict):
    """Generate detailed personalized recommendations"""
    st.subheader("💡 Personalized Learning Recommendations")

    recommendations = []

    # Performance-based recommendations
    avg_accuracy = quiz_metrics.get("avg_accuracy", 0)
    avg_speed = reading_metrics.get("avg_reading_speed", 0)
    avg_response_time = quiz_metrics.get("avg_response_time", 0)

    # Reading speed recommendations
    if avg_speed < 150:
        recommendations.append({
            "category": "Reading Speed",
            "priority": "High",
            "recommendation": "Focus on building reading fluency with easier texts",
            "action_items": [
                "Practice with short, familiar stories",
                "Use guided reading techniques",
                "Set daily reading time goals"
            ],
            "target": "150+ WPM"
        })
    elif avg_speed > 250:
        recommendations.append({
            "category": "Reading Speed",
            "priority": "Medium",
            "recommendation": "Balance speed with comprehension",
            "action_items": [
                "Slow down for complex passages",
                "Practice active reading strategies",
                "Focus on understanding over speed"
            ],
            "target": "Maintain speed while improving accuracy"
        })

    # Accuracy recommendations
    if avg_accuracy < 0.7:
        recommendations.append({
            "category": "Comprehension",
            "priority": "High",
            "recommendation": "Strengthen reading comprehension skills",
            "action_items": [
                "Read stories in your favorite topics",
                "Take notes while reading",
                "Practice summarization techniques"
            ],
            "target": "80%+ accuracy"
        })
    elif avg_accuracy > 0.9:
        recommendations.append({
            "category": "Challenge Level",
            "priority": "Medium",
            "recommendation": "Ready for more challenging content",
            "action_items": [
                "Try harder difficulty levels",
                "Explore new topic areas",
                "Set higher accuracy targets"
            ],
            "target": "Advanced level mastery"
        })

    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            with st.expander(f"{priority_colors[rec['priority']]} {rec['category']} - {rec['priority']} Priority"):
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Target:** {rec['target']}")
                st.write("**Action Items:**")
                for item in rec["action_items"]:
                    st.write(f"• {item}")
    else:
        st.success("🎉 Excellent performance! Keep up the great work across all areas.")

def create_goal_progress_tracking(performance_data: List[Dict], user_preferences: Dict):
    """Track progress towards user-defined learning goals"""
    st.subheader("🎯 Goal Progress Tracking")

    learning_goals = user_preferences.get("learning_goals", [])

    if not learning_goals:
        st.info("Set learning goals in your preferences to track progress!")
        return

    # Calculate current performance metrics
    if performance_data:
        current_accuracy = np.mean([s["accuracy_rate"] for s in performance_data[-5:]])  # Last 5 sessions
        current_speed = np.mean([s["reading_speed_wpm"] for s in performance_data[-5:]])

        # Define targets for each goal
        goal_targets = {
            "improve comprehension": {"current": current_accuracy, "target": 0.85, "unit": "%", "multiplier": 100},
            "increase speed": {"current": current_speed, "target": 200, "unit": " WPM", "multiplier": 1},
            "better vocabulary": {"current": current_accuracy, "target": 0.9, "unit": "%", "multiplier": 100},
            "critical thinking": {"current": current_accuracy, "target": 0.8, "unit": "%", "multiplier": 100},
            "enjoy reading": {"current": len(performance_data), "target": 20, "unit": " sessions", "multiplier": 1},
            "build confidence": {"current": current_accuracy, "target": 0.75, "unit": "%", "multiplier": 100}
        }

        cols = st.columns(min(len(learning_goals), 3))

        for i, goal in enumerate(learning_goals):
            col_idx = i % 3

            if goal in goal_targets:
                target_info = goal_targets[goal]
                current = target_info["current"]
                target = target_info["target"]
                unit = target_info["unit"]
                multiplier = target_info["multiplier"]

                progress = min(current / target, 1.0) if target > 0 else 0

                with cols[col_idx]:
                    st.markdown(f"**{goal.title()}**")
                    st.progress(progress)
                    current_display = current * multiplier
                    target_display = target * multiplier
                    st.write(f"{current_display:.1f} / {target_display:.1f}{unit}")

                    if progress >= 0.9:
                        st.success("🎉 Goal achieved!")
                    elif progress >= 0.7:
                        st.info("📈 Great progress!")
                    else:
                        st.warning("💪 Keep working!")
    else:
        st.info("Complete some sessions to track your goal progress!")

def create_future_predictions(performance_data: List[Dict]):
    """Create predictions for future performance"""
    st.subheader("🔮 Performance Predictions")

    if len(performance_data) < 3:
        st.info("Complete more sessions to unlock performance predictions!")
        return

    df = pd.DataFrame(performance_data)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.sort_values("session_date")
    df["session_number"] = range(1, len(df) + 1)

    # Predict next 5 sessions
    if len(df) >= 3:
        # Accuracy prediction
        accuracy_trend = np.polyfit(df["session_number"], df["accuracy_rate"], 1)
        speed_trend = np.polyfit(df["session_number"], df["reading_speed_wpm"], 1)

        future_sessions = list(range(len(df) + 1, len(df) + 6))
        predicted_accuracy = np.poly1d(accuracy_trend)(future_sessions)
        predicted_speed = np.poly1d(speed_trend)(future_sessions)

        # Ensure predictions are realistic
        predicted_accuracy = np.clip(predicted_accuracy, 0, 1)
        predicted_speed = np.clip(predicted_speed, 50, 500)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Accuracy Prediction**")
            current_accuracy = df["accuracy_rate"].iloc[-1]
            next_accuracy = predicted_accuracy[0]
            improvement = (next_accuracy - current_accuracy) * 100

            st.metric(
                "Next Session Prediction",
                f"{next_accuracy:.1%}",
                f"{improvement:+.1f}%"
            )

            if improvement > 2:
                st.success("📈 Expecting improvement!")
            elif improvement > -2:
                st.info("📊 Stable performance expected")
            else:
                st.warning("💪 Focus on improvement strategies")

        with col2:
            st.markdown("**🚀 Speed Prediction**")
            current_speed = df["reading_speed_wpm"].iloc[-1]
            next_speed = predicted_speed[0]
            speed_change = next_speed - current_speed

            st.metric(
                "Next Session Prediction",
                f"{next_speed:.0f} WPM",
                f"{speed_change:+.0f} WPM"
            )

            if speed_change > 5:
                st.success("🚀 Speed increasing!")
            elif speed_change > -5:
                st.info("📊 Steady speed expected")
            else:
                st.warning("🐌 Focus on reading fluency")

def ai_search_page():
    """AI-powered search and assistant page"""
    st.header("🤖 AI Learning Assistant & Smart Search")
    st.markdown("Ask questions, search for content, and get personalized learning recommendations!")

    # Initialize AI search engine
    if 'ai_search' not in st.session_state:
        with st.spinner("🤖 Initializing AI Assistant..."):
            st.session_state.ai_search = AISearchEngine()

    # Get user context
    user_context = {}
    if st.session_state.get('current_user'):
        user_context = {
            'preferences': st.session_state.db.get_user_preferences(st.session_state.current_user['user_id']),
            'performance': st.session_state.db.get_user_performance_data(st.session_state.current_user['user_id'], 10)
        }

    # Search interface
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "🔍 Ask me anything or search for content:",
            placeholder="e.g., 'Show me science stories', 'How to read faster?', 'What are good study tips?'",
            key="ai_search_query"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align button
        search_button = st.button("🚀 Search", type="primary")

    # Quick suggestion buttons
    st.markdown("**💡 Quick Questions:**")
    suggestion_cols = st.columns(4)

    suggestions = [
        "How to read faster?",
        #"Find adventure stories",
        "Improve comprehension",
        "Study tips for focus"
    ]

    suggestion_clicked = None
    for i, suggestion in enumerate(suggestions):
        if suggestion_cols[i].button(suggestion, key=f"suggestion_{i}"):
            suggestion_clicked = suggestion
            query = suggestion

    # Process query
    if (query and search_button) or suggestion_clicked or (query and len(query) > 3):
        with st.spinner("🤖 AI is analyzing your request..."):
            result = st.session_state.ai_search.natural_language_query(query, user_context)

            # Save query for analytics
            if st.session_state.get('current_user'):
                st.session_state.db.save_ai_search_query(
                    st.session_state.current_user['user_id'],
                    query,
                    result['intent'],
                    len(result.get('search_results', []))
                )

        # Display AI response
        if result['response']:
            st.markdown("### 🤖 AI Assistant Response")
            st.info(result['response'])

        # Display search results
        if result['search_results']:
            st.markdown("### 📚 Related Stories")
            for i, search_result in enumerate(result['search_results']):
                metadata = search_result['metadata']
                similarity = search_result['similarity']
                snippet = search_result['snippet']

                with st.expander(f"📖 {metadata['title']} (Relevance: {similarity:.1%})", expanded=(i==0)):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Excerpt:** {snippet}")
                        st.markdown(f"**Topics:** {', '.join(metadata['topics'])}")
                        st.markdown(f"**Category:** {metadata['category']}")

                    with col2:
                        st.markdown(f"**Difficulty:** {metadata['difficulty'].title()}")

                        if st.button(f"📖 Read This Story", key=f"read_{i}"):
                            # Navigate to reading page with this story
                            st.session_state.selected_story_from_ai = metadata['key']
                            st.success(f"🎯 Story selected: '{metadata['title']}' - Go to Reading & Quiz page to start!")

        # Display recommendations
        if result['recommendations']:
            st.markdown("### 💡 Personalized AI Recommendations")

            for i, rec in enumerate(result['recommendations']):
                priority_colors = {
                    'high': '🔴',
                    'medium': '🟡', 
                    'low': '🟢'
                }

                with st.expander(f"{priority_colors.get(rec['priority'], '🔵')} {rec['title']}", expanded=(i==0)):
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Recommended Action:** {rec['action']}")
                    st.markdown(f"**Type:** {rec['type'].replace('_', ' ').title()}")
                    st.markdown(f"**Priority:** {rec['priority'].title()}")

        # Display related suggestions
        if result['suggestions']:
            st.markdown("### 🎯 Related Questions")
            suggestion_cols = st.columns(2)

            for i, suggestion in enumerate(result['suggestions']):
                col_idx = i % 2
                if suggestion_cols[col_idx].button(suggestion, key=f"related_{i}"):
                    st.rerun()

    # AI Features Overview
    st.markdown("---")
    st.markdown("### 🌟 AI Assistant Capabilities")

    feature_cols = st.columns(3)

    with feature_cols[0]:
        st.markdown("""
        **🔍 Semantic Search**
        - Find stories by meaning, not just keywords
        - Understand context beyond literal terms
        - Get relevant content suggestions
        - Discover hidden connections
        """)

    with feature_cols[1]:
        st.markdown("""
        **🤖 Intelligent Assistant**
        - Get personalized study advice
        - Understand reading concepts
        - Improve comprehension skills
        - Tailored to your progress
        """)

    with feature_cols[2]:
        st.markdown("""
        **💡 Smart Recommendations**
        - Personalized learning paths
        - Performance-based suggestions
        - Goal-oriented guidance
        - Adaptive difficulty progression
        """)

    # Popular queries section based on user level
    if st.session_state.get('current_user'):
        st.markdown("---")
        st.markdown("### 🔥 Popular AI Queries")

        # Generate personalized popular queries
        performance_data = user_context.get('performance', [])
        if performance_data:
            avg_accuracy = np.mean([p.get('accuracy_rate', 0) for p in performance_data])

            if avg_accuracy < 0.7:
                popular_queries = [
                    "How to improve reading comprehension?",
                    "What are active reading strategies?",
                    "Find easy stories to practice with",
                    "Tips for better concentration while reading"
                ]
            elif avg_accuracy > 0.85:
                popular_queries = [
                    "Challenge me with harder stories",
                    "Advanced reading techniques",
                    "How to read critically?", 
                    "Find complex mystery stories"
                ]
            else:
                popular_queries = [
                    "How to balance speed and comprehension?",
                    "Find stories matching my interests",
                    "What's my optimal reading difficulty?",
                    "Study schedule recommendations"
                ]
        else:
            popular_queries = [
                "How do I get started with reading practice?",
                "What stories are best for beginners?", 
                "Basic reading improvement tips",
                "How to track my progress effectively?"
            ]

        query_cols = st.columns(2)
        for i, popular_query in enumerate(popular_queries):
            col_idx = i % 2
            if query_cols[col_idx].button(f"💭 {popular_query}", key=f"popular_{i}"):
                st.session_state.ai_search_query = popular_query
                st.rerun()

def init_session_state():
    """Initialize Streamlit session state with user management and AI features"""
    if 'db' not in st.session_state:
        st.session_state.db = PersonalizedLearningDB()

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LearningAnalyzer()

    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'current_story' not in st.session_state:
        st.session_state.current_story = None

    if 'reading_start_time' not in st.session_state:
        st.session_state.reading_start_time = None

    if 'reading_session' not in st.session_state:
        st.session_state.reading_session = None

    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None

    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0

    if 'question_start_time' not in st.session_state:
        st.session_state.question_start_time = None

    if 'quiz_responses' not in st.session_state:
        st.session_state.quiz_responses = []

    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False

def login_page():
    """FIXED user login interface with perfect color contrast"""
    st.title("🤖 AI-Enhanced Personalized Learning Platform")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # FIXED: Perfect readability with dark text on light background
        st.markdown("""
        <div style="
            background-color: #ffffff; 
            padding: 25px; 
            border-radius: 15px; 
            text-align: center; 
            border: 2px solid #007bff;
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
            margin-bottom: 25px;
        ">
            <h2 style="
                color: #2c3e50; 
                margin-bottom: 15px; 
                font-weight: 600;
                font-size: 24px;
            ">
                🌟 Welcome to Your AI Learning Journey
            </h2>
            <p style="
                color: #495057; 
                font-size: 16px; 
                margin: 0;
                line-height: 1.5;
            ">
                Enter your student ID to access AI-powered personalized learning
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show available users for demo
        all_users = st.session_state.db.get_all_users()

        st.markdown("### 👥 Available Student IDs:")
        users_df = pd.DataFrame(all_users)
        if not users_df.empty:
            users_df["last_login"] = pd.to_datetime(users_df["last_login"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(users_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Login form
        st.markdown("### 🔐 Student Login")

        with st.form("login_form"):
            user_id = st.text_input(
                "Student ID", 
                placeholder="Enter your ID (e.g., STU001)", 
                max_chars=10,
                help="Use one of the Student IDs from the table above"
            )

            submitted = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)

            if submitted:
                if not user_id.strip():
                    st.error("⚠️ Please enter your student ID")
                else:
                    user_data = st.session_state.db.authenticate_user(user_id.strip().upper())

                    if user_data:
                        st.session_state.current_user = user_data
                        st.session_state.logged_in = True
                        st.success(f"✅ Welcome back, {user_data['display_name']}!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid student ID. Please check and try again.")

        # Create new user option
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🆕 Create New Student Account"):
            st.markdown("**Don't have an account? Create one here:**")

            with st.form("create_user_form"):
                new_id = st.text_input("New Student ID", placeholder="STU006")
                new_name = st.text_input("Display Name", placeholder="Your Full Name")

                create_submitted = st.form_submit_button("✨ Create Account")

                if create_submitted:
                    if not new_id.strip() or not new_name.strip():
                        st.error("⚠️ Please fill in all fields")
                    else:
                        existing_user = st.session_state.db.authenticate_user(new_id.strip().upper())
                        if existing_user:
                            st.error("❌ Student ID already exists")
                        else:
                            try:
                                with sqlite3.connect(st.session_state.db.db_path) as conn:
                                    cursor = conn.cursor()
                                    default_prefs = {
                                        "preferred_topics": ["adventure", "science"],
                                        "preferred_difficulty": "medium",
                                        "preferred_story_length": "medium",
                                        "interests": ["reading", "learning"],
                                        "learning_goals": ["improve comprehension", "increase speed"]
                                    }

                                    cursor.execute(
                                        "INSERT INTO users (user_id, display_name, preferences) VALUES (?, ?, ?)",
                                        (new_id.strip().upper(), new_name.strip(), json.dumps(default_prefs))
                                    )
                                    conn.commit()

                                st.success(f"🎉 Account created! You can now login with {new_id.upper()}")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Error creating account: {str(e)}")

    # Instructions section with proper contrast
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="
            background-color: #e7f3ff; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 4px solid #007bff;
            margin-top: 20px;
        ">
            <h4 style="color: #0056b3; margin-bottom: 15px;">🎯 AI Features Include</h4>
            <ul style="margin: 0; padding-left: 20px; color: #2c3e50;">
                <li><strong>Smart Search:</strong> Find stories using natural language</li>
                <li><strong>AI Assistant:</strong> Get personalized learning advice</li>
                <li><strong>Progress Analytics:</strong> Advanced performance insights</li>
                <li><strong>Recommendations:</strong> AI-powered content suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def user_preferences_page():
    """FIXED User preferences management page - no more Streamlit API exceptions"""
    st.header(f"⚙️ Learning Preferences - {st.session_state.current_user['display_name']}")

    current_preferences = st.session_state.db.get_user_preferences(st.session_state.current_user["user_id"])

    # Create the preferences form
    with st.form("preferences_form", clear_on_submit=False):
        st.subheader("📚 Story Preferences")

        # Topic preferences
        available_topics = EnhancedContentProvider.get_available_topics()
        current_topics = current_preferences.get("preferred_topics", [])

        # FIXED: Filter current topics to ensure they exist in available options
        valid_topics = [topic for topic in current_topics if topic in available_topics]

        preferred_topics = st.multiselect(
            "Select your favorite topics:",
            options=available_topics,
            default=valid_topics,
            help="Choose topics you enjoy reading about - this helps our AI recommend better content!"
        )

        # Two-column layout for preferences
        col1, col2 = st.columns(2)

        with col1:
            # FIXED: Ensure difficulty is valid
            current_difficulty = current_preferences.get("preferred_difficulty", "medium")
            difficulty_options = ["easy", "medium", "hard"]
            if current_difficulty not in difficulty_options:
                current_difficulty = "medium"

            preferred_difficulty = st.selectbox(
                "Preferred difficulty level:",
                options=difficulty_options,
                index=difficulty_options.index(current_difficulty)
            )

            # FIXED: Ensure length is valid
            current_length = current_preferences.get("preferred_story_length", "medium")
            length_options = ["short", "medium", "long"]
            if current_length not in length_options:
                current_length = "medium"

            preferred_length = st.selectbox(
                "Preferred story length:",
                options=length_options,
                index=length_options.index(current_length)
            )

        with col2:
            # FIXED: Expanded interests options - now includes 'learning' and more
            interests_options = [
                "reading", "learning", "science", "technology", "history", 
                "nature", "adventure", "mystery", "fantasy", "creativity", 
                "problem solving", "storytelling", "research", "innovation"
            ]

            current_interests = current_preferences.get("interests", ["reading"])
            # FIXED: Filter current interests to ensure they exist in options
            valid_interests = [interest for interest in current_interests if interest in interests_options]
            if not valid_interests:  # If no valid interests, default to reading
                valid_interests = ["reading"]

            interests = st.multiselect(
                "General interests:",
                options=interests_options,
                default=valid_interests,
                help="Your interests help our AI assistant provide better recommendations"
            )

            # FIXED: Expanded learning goals options
            goals_options = [
                "improve comprehension", "increase speed", "better vocabulary", 
                "critical thinking", "enjoy reading", "build confidence",
                "analytical skills", "creative thinking", "focus improvement"
            ]

            current_goals = current_preferences.get("learning_goals", ["improve comprehension"])
            # FIXED: Filter current goals to ensure they exist in options
            valid_goals = [goal for goal in current_goals if goal in goals_options]
            if not valid_goals:  # If no valid goals, default to improve comprehension
                valid_goals = ["improve comprehension"]

            learning_goals = st.multiselect(
                "Learning goals:",
                options=goals_options,
                default=valid_goals,
                help="Goals help our AI track your progress and provide targeted advice"
            )

        st.subheader("🎯 Learning Style & AI Personalization")

        learning_style_info = st.text_area(
            "Tell us about your learning style (optional):",
            value=current_preferences.get("learning_style_notes", ""),
            placeholder="How do you learn best? What motivates you? This helps our AI provide better support.",
            height=100
        )

        # Add some spacing before the submit button
        st.markdown("---")

        # FIXED: Submit button - properly positioned and styled
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "💾 Save Preferences & Update AI", 
                type="primary",
                use_container_width=True,
                help="Save preferences to personalize your AI learning experience"
            )

        # Handle form submission
        if submitted:
            new_preferences = {
                "preferred_topics": preferred_topics,
                "preferred_difficulty": preferred_difficulty,
                "preferred_story_length": preferred_length,
                "interests": interests,
                "learning_goals": learning_goals,
                "learning_style_notes": learning_style_info
            }

            try:
                st.session_state.db.update_user_preferences(
                    st.session_state.current_user["user_id"], 
                    new_preferences
                )
                st.success("✅ Preferences saved! Your AI assistant is now more personalized.")
                st.balloons()  # Add celebration effect

                # Force a rerun to update the display
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error saving preferences: {str(e)}")

    # Show preference summary outside the form
    if current_preferences:
        st.markdown("---")
        st.subheader("📊 Your AI Personalization Profile")

        # Create three columns for summary display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📖 Favorite Topics:**")
            topics = current_preferences.get("preferred_topics", [])
            if topics:
                for topic in topics[:5]:  # Show first 5 topics
                    st.write(f"• {topic.title()}")
                if len(topics) > 5:
                    st.write(f"• ... and {len(topics) - 5} more")
            else:
                st.write("• No topics selected")

        with col2:
            st.markdown("**🎯 Learning Goals:**")
            goals = current_preferences.get("learning_goals", [])
            if goals:
                for goal in goals:
                    st.write(f"• {goal.title()}")
            else:
                st.write("• No goals selected")

        with col3:
            st.markdown("**⚙️ Settings:**")
            st.write(f"• **Difficulty:** {current_preferences.get('preferred_difficulty', 'medium').title()}")
            st.write(f"• **Length:** {current_preferences.get('preferred_story_length', 'medium').title()}")

            # Show number of interests
            interests_count = len(current_preferences.get("interests", []))
            st.write(f"• **Interests:** {interests_count} selected")

        # Show learning style notes if available
        style_notes = current_preferences.get("learning_style_notes", "")
        if style_notes:
            st.markdown("**📝 Learning Style Notes:**")
            st.info(style_notes)

        # AI Personalization tip
        st.info("🤖 **AI Tip:** Your preferences help our AI assistant provide more targeted recommendations and better search results!")

def reading_page():
    """Enhanced reading page with AI-powered story selection"""
    st.header(f"📖 AI-Enhanced Personalized Reading - {st.session_state.current_user['display_name']}")

    # Get user preferences
    user_preferences = st.session_state.db.get_user_preferences(st.session_state.current_user["user_id"])

    # Check if story was selected from AI search
    if st.session_state.get('selected_story_from_ai'):
        selected_story_key = st.session_state.selected_story_from_ai
        all_stories = EnhancedContentProvider.get_comprehensive_stories()
        available_stories = {selected_story_key: all_stories[selected_story_key]}
        st.info("🤖 **AI Selected Story:** This story was recommended by your AI assistant!")
        # Clear the selection
        del st.session_state.selected_story_from_ai
    else:
        # Get stories based on preferences
        available_stories = EnhancedContentProvider.get_stories_by_preferences(user_preferences)

    # Story selection with AI recommendations
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 AI-Recommended Stories for You")

        # Show why these stories are recommended
        preferred_topics = user_preferences.get("preferred_topics", [])
        if preferred_topics:
            st.info(f"📚 Stories selected by AI based on your interests: {', '.join(preferred_topics)}")

        # Story selection
        story_options = {key: f"{data['title']} ({data['difficulty'].title()}) - {', '.join(data['topics'])}" 
                        for key, data in available_stories.items()}

        selected_story_key = st.selectbox(
            "Choose a story to read:",
            options=list(story_options.keys()),
            format_func=lambda x: story_options[x]
        )

    with col2:
        # Get performance data for AI recommendations
        performance_data = st.session_state.db.get_user_performance_data(st.session_state.current_user["user_id"], 10)
        recommended_level = st.session_state.analyzer.recommend_difficulty_level(performance_data)

        st.markdown("### 🎯 Your AI Learning Status")
        if performance_data:
            latest_accuracy = performance_data[0]["accuracy_rate"]
            latest_speed = performance_data[0]["reading_speed_wpm"]
            st.metric("Latest Accuracy", f"{latest_accuracy:.1%}")
            st.metric("Reading Speed", f"{latest_speed:.1f} WPM")

        st.info(f"🤖 **AI Recommended Level:** {recommended_level.title()}")

        # AI-powered personalized tips
        st.markdown("### 💡 AI Personalized Tips")
        tips = []
        if "increase speed" in user_preferences.get("learning_goals", []):
            tips.append("🏃‍♂️ Focus on smooth reading flow")
        if "improve comprehension" in user_preferences.get("learning_goals", []):
            tips.append("🧠 Take time to understand details")
        if "enjoy reading" in user_preferences.get("learning_goals", []):
            tips.append("😊 Choose topics you love!")

        # Add AI-generated tip based on performance
        if performance_data:
            avg_accuracy = np.mean([p.get('accuracy_rate', 0) for p in performance_data])
            if avg_accuracy < 0.7:
                tips.append("🔍 Try our AI Assistant for comprehension help!")
            elif avg_accuracy > 0.85:
                tips.append("🚀 Ready for AI-recommended challenges!")

        for tip in tips:
            st.write(tip)

        # Link to AI assistant
        st.markdown("---")
        if st.button("🤖 Ask AI Assistant", type="secondary", use_container_width=True):
            st.info("💡 Go to 'AI Search & Assistant' tab to get personalized help!")

    if st.button("📚 Start AI-Tracked Reading Session", type="primary"):
        story_data = available_stories[selected_story_key]
        st.session_state.current_story = story_data
        st.session_state.reading_start_time = time.time()
        st.session_state.quiz_completed = False
        st.rerun()

    # Display story and track reading (enhanced with AI features)
    if st.session_state.current_story and st.session_state.reading_start_time:
        story = st.session_state.current_story

        # Reading timer
        if not st.session_state.reading_session:
            elapsed_time = time.time() - st.session_state.reading_start_time
            st.info(f"⏱️ Reading time: {elapsed_time:.1f} seconds (AI is tracking your progress)")

        # Story content with AI enhancements
        st.subheader(f"📖 {story['title']}")

        # Show AI-matched story topics
        story_topics = story.get("topics", [])
        user_topics = set(user_preferences.get("preferred_topics", []))
        matching_topics = set(story_topics).intersection(user_topics)

        if matching_topics:
            st.success(f"✨ **AI Match:** This story matches your interests: {', '.join(matching_topics)}")

        with st.container():
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 25px; border-radius: 10px; 
                       border-left: 4px solid #007bff; line-height: 1.6; font-size: 16px;">
                {story['content'].strip()}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Finish reading button with AI enhancement
        if not st.session_state.reading_session:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("✅ Finished Reading - Start AI-Enhanced Quiz", type="primary", use_container_width=True):
                    # Calculate reading metrics
                    reading_end_time = time.time()
                    reading_duration = reading_end_time - st.session_state.reading_start_time
                    word_count = len(story["content"].split())
                    reading_speed_wpm = (word_count / reading_duration) * 60

                    # Create reading session
                    st.session_state.reading_session = ReadingSession(
                        story_id=f"story_{selected_story_key}_{int(time.time())}",
                        story_content=story["content"],
                        reading_start_time=st.session_state.reading_start_time,
                        reading_end_time=reading_end_time,
                        reading_duration=reading_duration,
                        word_count=word_count,
                        reading_speed_wpm=reading_speed_wpm
                    )

                    # Prepare quiz data
                    st.session_state.quiz_data = story["questions"]
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_responses = []

                    st.success(f"📊 AI Analysis: Reading speed {reading_speed_wpm:.1f} WPM")
                    st.rerun()

        # Quiz section
        if st.session_state.reading_session and st.session_state.quiz_data and not st.session_state.quiz_completed:
            display_quiz()

def display_quiz():
    """Display quiz with detailed time tracking and AI features"""
    st.subheader("🧠 AI-Enhanced Comprehension Quiz")

    questions = st.session_state.quiz_data
    current_index = st.session_state.current_question_index

    if current_index < len(questions):
        question = questions[current_index]

        # Start question timer
        if st.session_state.question_start_time is None:
            st.session_state.question_start_time = time.time()

        # Display progress with AI enhancement
        progress = (current_index + 1) / len(questions)
        st.progress(progress)
        st.write(f"Question {current_index + 1} of {len(questions)} (AI is analyzing your responses)")

        # Question timer
        elapsed = time.time() - st.session_state.question_start_time
        st.write(f"⏱️ Time on this question: {elapsed:.1f} seconds")

        # Question content
        st.write(f"**{question['text']}**")

        # Answer options
        answer = st.radio(
            "Select your answer:",
            question["options"],
            key=f"q_{current_index}"
        )

        # Next button
        if st.button("➡️ Next Question" if current_index < len(questions) - 1 else "🏁 Finish AI Analysis"):
            # Record response
            response_time = time.time() - st.session_state.question_start_time
            is_correct = answer == question["correct"]

            response = QuestionResponse(
                question_id=f"q_{current_index}",
                question_text=question["text"],
                options=question["options"],
                correct_answer=question["correct"],
                user_answer=answer,
                is_correct=is_correct,
                response_time=response_time,
                difficulty_level=st.session_state.current_story.get("difficulty", "medium"),
                question_type=question.get("type", "general")
            )

            st.session_state.quiz_responses.append(response)
            st.session_state.current_question_index += 1
            st.session_state.question_start_time = None

            if current_index == len(questions) - 1:
                # Quiz completed
                complete_quiz_session()

            st.rerun()

def complete_quiz_session():
    """Complete quiz session and save results with AI analysis"""
    responses = st.session_state.quiz_responses

    # Calculate quiz metrics
    total_score = sum(1 for r in responses if r.is_correct)
    accuracy_rate = total_score / len(responses)
    average_response_time = sum(r.response_time for r in responses) / len(responses)

    # Create quiz session
    session_id = f"quiz_{st.session_state.current_user['user_id']}_{int(time.time())}_{random.randint(1000, 9999)}"
    quiz_session = QuizSession(
        session_id=session_id,
        story_id=st.session_state.reading_session.story_id,
        reading_session=st.session_state.reading_session,
        responses=responses,
        total_score=total_score,
        total_questions=len(responses),
        accuracy_rate=accuracy_rate,
        average_response_time=average_response_time,
        session_date=datetime.now(),
        difficulty_level=st.session_state.current_story.get("difficulty", "medium")
    )

    # Save to database with user ID
    story_id = st.session_state.db.save_story(
        st.session_state.current_story["title"],
        st.session_state.current_story["content"],
        st.session_state.current_story.get("difficulty", "medium"),
        st.session_state.current_story["category"],
        st.session_state.current_story.get("topics", [])
    )

    reading_session_id = st.session_state.db.save_reading_session(
        st.session_state.reading_session, 
        story_id,
        st.session_state.current_user["user_id"]
    )

    st.session_state.db.save_quiz_session(
        quiz_session, story_id, reading_session_id, 
        st.session_state.current_user["user_id"]
    )

    st.session_state.quiz_completed = True

    # Display results with AI analysis
    display_quiz_results(quiz_session)

def display_quiz_results(quiz_session: QuizSession):
    """Display detailed quiz results with AI insights"""
    st.success("🎉 Quiz Completed! AI Analysis Ready")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Score", f"{quiz_session.total_score}/{quiz_session.total_questions}")

    with col2:
        st.metric("Accuracy", f"{quiz_session.accuracy_rate:.1%}")

    with col3:
        st.metric("Avg Response Time", f"{quiz_session.average_response_time:.1f}s")

    with col4:
        st.metric("Reading Speed", f"{quiz_session.reading_session.reading_speed_wpm:.1f} WPM")

    # AI-Enhanced Performance feedback
    user_preferences = st.session_state.db.get_user_preferences(st.session_state.current_user["user_id"])

    st.markdown("### 🤖 AI Performance Analysis")

    if quiz_session.accuracy_rate >= 0.8:
        if user_preferences.get("preferred_difficulty") != "hard":
            st.success("🎯 **AI Recommendation:** Excellent performance! Consider trying harder difficulty levels.")
        else:
            st.success("🌟 **AI Analysis:** Outstanding work at the hard difficulty level!")
    elif quiz_session.accuracy_rate >= 0.6:
        st.info("👍 **AI Feedback:** Good job! Keep practicing with your favorite topics to improve.")
    else:
        favorite_topics = user_preferences.get("preferred_topics", [])
        st.warning(f"💪 **AI Suggestion:** Keep practicing! Try stories about {', '.join(favorite_topics[:2])} - topics you enjoy.")

    # Link to AI assistant
    st.info("🤖 **Pro Tip:** Ask our AI Assistant for personalized improvement strategies based on this performance!")

    # Detailed question analysis
    st.subheader("📊 Question-by-Question AI Analysis")

    for i, response in enumerate(quiz_session.responses):
        with st.expander(f"Question {i+1} - {'✅ Correct' if response.is_correct else '❌ Incorrect'} ({response.response_time:.1f}s)"):
            st.write(f"**Question:** {response.question_text}")
            st.write(f"**Your Answer:** {response.user_answer}")
            st.write(f"**Correct Answer:** {response.correct_answer}")
            st.write(f"**Response Time:** {response.response_time:.1f} seconds")
            st.write(f"**Question Type:** {response.question_type}")

            # AI insight for this question
            if not response.is_correct:
                st.info(f"🤖 **AI Tip:** This was a {response.question_type} question. Try focusing on this skill type in future reading.")

def analytics_page():
    """User-specific analytics dashboard with AI insights"""
    st.header(f"📊 AI-Enhanced Learning Analytics - {st.session_state.current_user['display_name']}")

    # Get user-specific performance data
    performance_data = st.session_state.db.get_user_performance_data(
        st.session_state.current_user["user_id"], 50
    )
    user_preferences = st.session_state.db.get_user_preferences(st.session_state.current_user["user_id"])

    if not performance_data:
        st.info("📈 No performance data available yet. Complete some reading sessions to see AI-powered analytics!")
        return

    # Calculate metrics
    reading_metrics = st.session_state.analyzer.calculate_reading_metrics(performance_data)
    quiz_metrics = st.session_state.analyzer.calculate_quiz_metrics(performance_data)

    # Summary metrics with AI enhancement
    st.subheader("🎯 Your AI-Analyzed Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Reading Speed", 
            f"{reading_metrics.get('avg_reading_speed', 0):.1f} WPM",
            f"{reading_metrics.get('reading_speed_trend', 0):+.1f}"
        )

    with col2:
        st.metric(
            "Accuracy", 
            f"{quiz_metrics.get('avg_accuracy', 0):.1%}",
            f"{quiz_metrics.get('accuracy_trend', 0):+.1%}"
        )

    with col3:
        st.metric(
            "Response Time", 
            f"{quiz_metrics.get('avg_response_time', 0):.1f}s",
            f"{quiz_metrics.get('response_time_trend', 0):+.1f}s"
        )

    with col4:
        st.metric("Total Sessions", len(performance_data))

    with col5:
        consistency = min(reading_metrics.get("reading_consistency", 0), quiz_metrics.get("consistency_score", 0))
        st.metric("AI Consistency Score", f"{consistency:.1%}")

    # AI-Enhanced Personalized insights
    st.subheader("🤖 AI Personalized Learning Insights")
    insights = st.session_state.analyzer.generate_personalized_insights(
        reading_metrics, quiz_metrics, performance_data, user_preferences
    )

    for insight in insights:
        st.info(f"🔍 {insight}")

    # Add link to AI assistant
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🤖 Get More AI Insights & Recommendations", type="primary", use_container_width=True):
            st.info("💡 Visit the 'AI Search & Assistant' tab for detailed personalized advice!")

    # User progress towards goals with AI tracking
    st.subheader("🎯 AI-Tracked Goal Progress")
    learning_goals = user_preferences.get("learning_goals", [])

    if not learning_goals:
        st.info("🤖 Set learning goals in your preferences to enable AI-powered goal tracking!")
        return

    for goal in learning_goals:
        if goal == "improve comprehension":
            current_accuracy = quiz_metrics.get("avg_accuracy", 0)
            target_accuracy = 0.85
            progress = min(current_accuracy / target_accuracy, 1.0)
            st.progress(progress)
            st.write(f"**{goal.title()}:** {current_accuracy:.1%} / {target_accuracy:.1%} target")

            # AI recommendation
            if progress >= 0.9:
                st.success("🤖 **AI:** Comprehension goal achieved! Ready for advanced challenges.")
            elif progress >= 0.5:
                st.info("🤖 **AI:** Good progress! Keep practicing with your favorite topics.")
            else:
                st.warning("🤖 **AI:** Try our AI Assistant for comprehension improvement strategies.")

        elif goal == "increase speed":
            current_speed = reading_metrics.get("avg_reading_speed", 0)
            target_speed = 200
            progress = min(current_speed / target_speed, 1.0)
            st.progress(progress)
            st.write(f"**{goal.title()}:** {current_speed:.1f} / {target_speed} WPM target")

            # AI recommendation
            if progress >= 0.9:
                st.success("🤖 **AI:** Speed goal achieved! Maintaining comprehension is now key.")
            elif progress >= 0.5:
                st.info("🤖 **AI:** Steady progress! Practice with familiar topics to build fluency.")
            else:
                st.warning("🤖 **AI:** Ask our AI Assistant for speed reading techniques.")

def main():
    """Main application function with AI-enhanced features"""
    init_session_state()

    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
        return

    # Main app interface
    user = st.session_state.current_user

    # Sidebar with user info and AI features
    st.sidebar.title(f"👋 Welcome {user['display_name']}")
    st.sidebar.markdown(f"**ID:** {user['user_id']}")
    st.sidebar.markdown("---")

    # Enhanced navigation with AI Search
    page = st.sidebar.selectbox(
        "🧭 Navigate to:",
        [
            "📖 Reading & Quiz", 
            "📊 Learning Analytics", 
            "⚙️ Preferences", 
            "🎯 Progress Insights",
            "🤖 AI Search & Assistant"  # AI FEATURE
        ]
    )

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.current_user = None
        st.session_state.logged_in = False
        st.session_state.current_story = None
        st.session_state.reading_session = None
        st.session_state.quiz_data = None
        # Clear AI search state
        if 'ai_search' in st.session_state:
            del st.session_state.ai_search
        st.rerun()

    st.sidebar.markdown("---")

    # Quick user stats with AI insights
    performance_data = st.session_state.db.get_user_performance_data(user["user_id"], 5)

    if performance_data:
        latest_session = performance_data[0]
        st.sidebar.metric("Latest Score", f"{latest_session['accuracy_rate']:.1%}")
        st.sidebar.metric("Reading Speed", f"{latest_session['reading_speed_wpm']:.1f} WPM")
        st.sidebar.metric("Sessions Completed", len(performance_data))

        # AI-powered quick insights in sidebar
        st.sidebar.markdown("**🤖 AI Quick Insights:**")
        recent_accuracy = latest_session['accuracy_rate']
        if recent_accuracy > 0.9:
            st.sidebar.success("🌟 Excellent work! Ready for harder challenges?")
        elif recent_accuracy < 0.6:
            st.sidebar.info("💡 Try the AI Assistant for comprehension tips!")
        else:
            st.sidebar.info("📈 Steady progress! Keep it up!")
    else:
        st.sidebar.info("Complete your first session!")
        st.sidebar.markdown("**🤖 AI Ready:** Your AI assistant is waiting to help!")

    st.sidebar.markdown("---")

    # User preferences summary with AI enhancement
    user_preferences = st.session_state.db.get_user_preferences(user["user_id"])
    preferred_topics = user_preferences.get("preferred_topics", [])

    if preferred_topics:
        st.sidebar.markdown("**📚 Your AI Profile:**")
        for topic in preferred_topics[:3]:  # Show top 3
            st.sidebar.write(f"• {topic.title()}")

        # AI recommendation
        if len(preferred_topics) < 3:
            st.sidebar.info("🤖 Add more interests for better AI recommendations!")

    # Route to pages
    if page == "📖 Reading & Quiz":
        reading_page()
    elif page == "📊 Learning Analytics":
        analytics_page()
    elif page == "⚙️ Preferences":
        user_preferences_page()
    elif page == "🎯 Progress Insights":
        progress_insights_page()
    elif page == "🤖 AI Search & Assistant":
        ai_search_page()  # NEW AI SEARCH PAGE

if __name__ == "__main__":
    main()
