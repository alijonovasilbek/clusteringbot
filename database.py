# database.py
import sqlite3
import json
from datetime import datetime
import config


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Ma'lumotlar bazasi jadvallarini yaratish"""
        cursor = self.conn.cursor()

        # Foydalanuvchilar jadvali
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_analyses INTEGER DEFAULT 0
            )
        ''')

        # Tahlillar jadvali
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                algorithm TEXT,
                dataset_name TEXT,
                parameters TEXT,
                n_clusters INTEGER,
                n_noise_points INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Default datasetlar
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS default_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                n_samples INTEGER,
                n_features INTEGER,
                data_json TEXT
            )
        ''')

        self.conn.commit()
        self._insert_default_datasets()

    def _insert_default_datasets(self):
        """Default datasetlarni kiritish"""
        import numpy as np
        from sklearn.datasets import make_blobs, make_moons, make_circles

        datasets = [
            {
                'name': 'Sferik Klasterlar',
                'description': '3 ta dumaloq shakldagi klaster',
                'generator': lambda: make_blobs(n_samples=300, centers=3, n_features=2,
                                                cluster_std=0.8, random_state=42)
            },
            {
                'name': 'Yarim Oy',
                'description': '2 ta yarim oy shaklidagi klaster',
                'generator': lambda: make_moons(n_samples=300, noise=0.05, random_state=42)
            },
            {
                'name': 'Doiralar',
                'description': 'Ichki va tashqi doira shaklidagi klasterlar',
                'generator': lambda: make_circles(n_samples=300, noise=0.05,
                                                  factor=0.5, random_state=42)
            },
            {
                'name': 'Tasodifiy Nuqtalar',
                'description': 'Tasodifiy tarqalgan 500 ta nuqta',
                'generator': lambda: (np.random.rand(500, 2) * 10, None)
            }
        ]

        cursor = self.conn.cursor()

        for ds in datasets:
            X, _ = ds['generator']()
            data_json = json.dumps(X.tolist())

            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO default_datasets 
                    (name, description, n_samples, n_features, data_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (ds['name'], ds['description'], len(X), X.shape[1], data_json))
            except:
                pass

        self.conn.commit()

    def add_user(self, user_id, username, first_name, last_name):
        """Yangi foydalanuvchi qo'shish"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO users (user_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
        ''', (user_id, username, first_name, last_name))
        self.conn.commit()

    def add_analysis(self, user_id, algorithm, dataset_name, parameters,
                     n_clusters, n_noise_points=0):
        """Tahlil natijasini saqlash"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO analyses 
            (user_id, algorithm, dataset_name, parameters, n_clusters, n_noise_points)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, algorithm, dataset_name, json.dumps(parameters),
              n_clusters, n_noise_points))

        cursor.execute('''
            UPDATE users SET total_analyses = total_analyses + 1 
            WHERE user_id = ?
        ''', (user_id,))

        self.conn.commit()

    def get_user_stats(self, user_id):
        """Foydalanuvchi statistikasi"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT total_analyses, join_date FROM users WHERE user_id = ?
        ''', (user_id,))
        return cursor.fetchone()

    def get_user_history(self, user_id, limit=10):
        """Foydalanuvchi tarixi"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT algorithm, dataset_name, n_clusters, created_at 
            FROM analyses 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        return cursor.fetchall()

    def get_default_datasets(self):
        """Default datasetlarni olish"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT name, description FROM default_datasets')
        return cursor.fetchall()

    def get_dataset_by_name(self, name):
        """Dataset ma'lumotlarini olish"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT data_json FROM default_datasets WHERE name = ?', (name,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None