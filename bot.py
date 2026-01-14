# bot.py
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, filters, ContextTypes
)
import numpy as np
import pandas as pd
import os

from database import Database
from clustering_engine import KMeans, DBSCAN, ElbowMethod
from visualizer import Visualizer
import config

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Database
db = Database()

# Conversation states
(CHOOSING_ALGORITHM, CHOOSING_DATASET, CHOOSING_SOURCE,
 UPLOADING_FILE, KMEANS_K, KMEANS_CONFIRM,
 DBSCAN_EPS, DBSCAN_MINPTS, DBSCAN_CONFIRM) = range(9)


class ClusteringBot:

    def __init__(self):
        self.viz = Visualizer()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start komandasi"""
        user = update.effective_user

        # Foydalanuvchini bazaga qo'shish
        db.add_user(user.id, user.username, user.first_name, user.last_name)

        welcome_text = f"""
🤖 <b>Clustering Bot'ga Xush Kelibsiz!</b>

Salom {user.first_name}! 👋

Men sizga <b>K-Means</b> va <b>DBSCAN</b> clustering algoritmlarini ishlatishda yordam beraman.

<b>📊 Imkoniyatlar:</b>
- Default datasetlar bilan ishlash
- O'z ma'lumotlaringizni yuklash (CSV, Excel)
- Interaktiv parametrlar sozlash
- Professional grafiklar olish
- Tahlil tarixini ko'rish
- Algoritmllarni taqqoslash

<b>📝 Komandalar:</b>
/analyze - Yangi tahlil boshlash
/history - Tahlillar tarixi
/stats - Statistika
/help - Yordam
/about - Bot haqida

Boshlash uchun /analyze ni bosing! 🚀
        """

        await update.message.reply_text(
            welcome_text,
            parse_mode='HTML'
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Yordam"""
        help_text = """
📖 <b>Yordam Bo'limi</b>

<b>🔹 K-Means Algoritmi:</b>
Ma'lumotlarni K ta klasterga ajratadi. Siz K qiymatini belgilaysiz.
Optimal: Dumaloq shakldagi klasterlar

<b>Parametrlar:</b>
- K - Klasterlar soni (1-10)
- Max Iterations - Maksimal iteratsiyalar

<b>🔹 DBSCAN Algoritmi:</b>
Zichlikka asoslangan klasterlash. Shovqinlarni avtomatik topadi.
Optimal: Har qanday shakldagi klasterlar

<b>Parametrlar:</b>
- Epsilon - Qo'shni radiusi (0.1-2.0)
- MinPts - Minimum nuqtalar (3-20)

<b>📁 Fayl Yuklash:</b>
- CSV yoki Excel formatda
- Maksimal 10MB
- Kamida 2 ta ustun kerak
- Birinchi 2 ta ustun ishlatiladi

<b>❓ Savollar bo'lsa:</b>
@your_support_username ga murojaat qiling
        """

        await update.message.reply_text(help_text, parse_mode='HTML')

    async def about_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bot haqida"""
        about_text = """
ℹ️ <b>Bot Haqida</b>

<b>Versiya:</b> 2.0
<b>Muallif:</b> @your_username
<b>Yaratilgan:</b> 2025

<b>Texnologiyalar:</b>
- Python 3.11
- python-telegram-bot
- NumPy, Pandas
- Matplotlib, Seaborn
- SQLite

<b>GitHub:</b> github.com/your_repo

<b>Feedback:</b>
Fikr-mulohazalaringizni yuboring! 💬
        """

        await update.message.reply_text(about_text, parse_mode='HTML')

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tahlilni boshlash"""
        keyboard = [
            [InlineKeyboardButton("🎯 K-Means", callback_data='algo_kmeans')],
            [InlineKeyboardButton("🌐 DBSCAN", callback_data='algo_dbscan')],
            [InlineKeyboardButton("⚖️ Ikkalasini Taqqoslash", callback_data='algo_compare')],
            [InlineKeyboardButton("❌ Bekor qilish", callback_data='cancel')]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🤔 <b>Qaysi algoritmdan foydalanmoqchisiz?</b>",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return CHOOSING_ALGORITHM

    async def algorithm_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Algoritm tanlandi"""
        query = update.callback_query
        await query.answer()

        choice = query.data.split('_')[1]
        context.user_data['algorithm'] = choice

        keyboard = [
            [InlineKeyboardButton("📦 Default Dataset", callback_data='source_default')],
            [InlineKeyboardButton("📤 O'z Faylimni Yuklash", callback_data='source_upload')],
            [InlineKeyboardButton("🔙 Ortga", callback_data='back_to_algo')]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "📊 <b>Ma'lumot manbai:</b>",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return CHOOSING_SOURCE

    async def source_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manba tanlandi"""
        query = update.callback_query
        await query.answer()

        choice = query.data.split('_')[1]

        if choice == 'default':
            # Default datasetlarni ko'rsatish
            datasets = db.get_default_datasets()

            keyboard = []
            for name, desc in datasets:
                keyboard.append([InlineKeyboardButton(
                    f"📊 {name}",
                    callback_data=f'dataset_{name}'
                )])
            keyboard.append([InlineKeyboardButton("🔙 Ortga", callback_data='back_to_source')])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "🗂 <b>Dataset tanlang:</b>",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )

            return CHOOSING_DATASET

        else:  # upload
            await query.edit_message_text(
                "📤 <b>CSV yoki Excel faylni yuboring:</b>\n\n"
                "📋 Talablar:\n"
                "• Maksimal hajm: 10 MB\n"
                "• Format: CSV yoki XLSX\n"
                "• Kamida 2 ta ustun\n"
                "• Maksimal 10,000 qator\n\n"
                "Bekor qilish uchun /cancel",
                parse_mode='HTML'
            )

            return UPLOADING_FILE

    async def dataset_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Dataset tanlandi"""
        query = update.callback_query
        await query.answer()

        dataset_name = query.data.replace('dataset_', '')
        context.user_data['dataset_name'] = dataset_name

        # Datasetni yuklash
        data = db.get_dataset_by_name(dataset_name)
        context.user_data['data'] = np.array(data)

        await query.edit_message_text(
            f"✅ Dataset tanlandi: <b>{dataset_name}</b>\n"
            f"📊 Nuqtalar soni: {len(data)}\n\n"
            "⏳ Parametrlarni sozlang...",
            parse_mode='HTML'
        )

        # Algoritmga qarab keyingi qadamga o'tish
        return await self.setup_algorithm_params(update, context)

    async def file_uploaded(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Fayl yuklandi"""
        file = update.message.document

        # Fayl hajmini tekshirish
        if file.file_size > config.MAX_FILE_SIZE:
            await update.message.reply_text(
                "❌ Fayl hajmi juda katta! (Maks: 10 MB)"
            )
            return UPLOADING_FILE

        # Fayl formatini tekshirish
        file_ext = file.file_name.split('.')[-1].lower()
        if file_ext not in ['csv', 'xlsx', 'xls']:
            await update.message.reply_text(
                "❌ Noto'g'ri format! Faqat CSV yoki Excel yuklang."
            )
            return UPLOADING_FILE

        # Faylni yuklab olish
        new_file = await file.get_file()
        file_path = os.path.join(config.UPLOAD_FOLDER, f"{update.effective_user.id}_{file.file_name}")
        await new_file.download_to_drive(file_path)

        try:
            # Faylni o'qish
            if file_ext == 'csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Tekshirish
            if len(df.columns) < 2:
                await update.message.reply_text(
                    "❌ Kamida 2 ta ustun bo'lishi kerak!"
                )
                return UPLOADING_FILE

            if len(df) > config.MAX_ROWS:
                await update.message.reply_text(
                    f"❌ Juda ko'p qator! (Maks: {config.MAX_ROWS})"
                )
                return UPLOADING_FILE

            # Faqat birinchi 2 ta raqamli ustunni olish
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]

            if len(numeric_cols) < 2:
                await update.message.reply_text(
                    "❌ Kamida 2 ta raqamli ustun bo'lishi kerak!"
                )
                return UPLOADING_FILE

            X = df[numeric_cols].values

            context.user_data['data'] = X
            context.user_data['dataset_name'] = file.file_name

            await update.message.reply_text(
                f"✅ <b>Fayl yuklandi!</b>\n\n"
                f"📊 Qatorlar: {len(X)}\n"
                f"📈 Ustunlar: {', '.join(numeric_cols)}\n\n"
                "⏳ Parametrlarni sozlang...",
                parse_mode='HTML'
            )

            # Faylni o'chirish
            os.remove(file_path)

            # Parametrlarni sozlash
            return await self.setup_algorithm_params(update, context)

        except Exception as e:
            logger.error(f"Fayl o'qishda xato: {e}")
            await update.message.reply_text(
                "❌ Faylni o'qishda xatolik yuz berdi!"
            )
            return UPLOADING_FILE

    async def setup_algorithm_params(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Algoritm parametrlarini sozlash"""
        algorithm = context.user_data.get('algorithm')
        X = context.user_data.get('data')

        if algorithm == 'kmeans':
            # Elbow method
            await self.send_typing(update, context)
            k_range, inertias = ElbowMethod.calculate(X, max_k=10)

            # Elbow grafigini yuborish
            elbow_img = self.viz.plot_elbow(k_range, inertias)

            if update.callback_query:
                await update.callback_query.message.reply_photo(
                    photo=elbow_img,
                    caption="📊 <b>Elbow Method</b>\n\n"
                            "Optimal K ni tanlash uchun 'tirsak' nuqtasini qidiring!",
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_photo(
                    photo=elbow_img,
                    caption="📊 <b>Elbow Method</b>\n\n"
                            "Optimal K ni tanlash uchun 'tirsak' nuqtasini qidiring!",
                    parse_mode='HTML'
                )

            # K ni tanlash
            keyboard = []
            for k in range(2, min(11, len(X))):
                keyboard.append([InlineKeyboardButton(f"K = {k}", callback_data=f'k_{k}')])

            reply_markup = InlineKeyboardMarkup(keyboard)

            if update.callback_query:
                await update.callback_query.message.reply_text(
                    "🔢 <b>K qiymatini tanlang (Klasterlar soni):</b>",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(
                    "🔢 <b>K qiymatini tanlang (Klasterlar soni):</b>",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )

            return KMEANS_K

        elif algorithm == 'dbscan':
            keyboard = []
            # Epsilon qiymatlari
            for eps in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
                keyboard.append([InlineKeyboardButton(
                    f"ε = {eps}",
                    callback_data=f'eps_{eps}'
                )])
            keyboard.append([InlineKeyboardButton("✏️ Boshqa qiymat", callback_data='eps_custom')])

            reply_markup = InlineKeyboardMarkup(keyboard)

            if update.callback_query:
                await update.callback_query.message.reply_text(
                    "📏 <b>Epsilon (ε) qiymatini tanlang:</b>\n\n"
                    "Bu qo'shni nuqtalar orasidagi maksimal masofa.",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(
                    "📏 <b>Epsilon (ε) qiymatini tanlang:</b>\n\n"
                    "Bu qo'shni nuqtalar orasidagi maksimal masofa.",
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )

            return DBSCAN_EPS

        else:  # compare
            # Ikkalasini ham ishlatish
            await self.run_comparison(update, context)
            return ConversationHandler.END

    async def kmeans_k_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """K tanlandi"""
        query = update.callback_query
        await query.answer()

        k = int(query.data.split('_')[1])
        context.user_data['k'] = k

        keyboard = [
            [InlineKeyboardButton("✅ Tahlilni Boshlash", callback_data='confirm_yes')],
            [InlineKeyboardButton("🔙 Ortga", callback_data='confirm_no')]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"📋 <b>Tahlil Sozlamalari:</b>\n\n"
            f"🎯 Algoritm: K-Means\n"
            f"📊 Dataset: {context.user_data.get('dataset_name')}\n"
            f"🔢 K (Klasterlar): {k}\n"
            f"📈 Iteratsiyalar: {config.DEFAULT_KMEANS_ITERATIONS}\n\n"
            "Davom etamizmi?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return KMEANS_CONFIRM

    async def kmeans_confirmed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """K-Means tahlilni boshlash"""
        query = update.callback_query
        await query.answer()

        if query.data == 'confirm_no':
            await query.edit_message_text("❌ Bekor qilindi.")
            return ConversationHandler.END

        await query.edit_message_text("⏳ <b>Tahlil boshlanmoqda...</b>", parse_mode='HTML')

        # Ma'lumotlarni olish
        X = context.user_data.get('data')
        k = context.user_data.get('k')

        # K-Means
        kmeans = KMeans(k=k, max_iters=config.DEFAULT_KMEANS_ITERATIONS, random_state=42)
        kmeans.fit(X)

        # Grafik
        img = self.viz.plot_kmeans(X, kmeans, f"K-Means (K={k})")

        # Klaster ma'lumotlari
        cluster_info = kmeans.get_cluster_info()
        info_text = "📊 <b>Klaster Ma'lumotlari:</b>\n\n"

        for cluster in cluster_info:
            info_text += (
                f"🔹 <b>Klaster {cluster['cluster_id']}</b>\n"
                f"   • Nuqtalar: {cluster['n_points']}\n"
                f"   • Foiz: {cluster['percentage']:.1f}%\n\n"
            )

        info_text += (
            f"📈 <b>Umumiy:</b>\n"
            f"   • Iteratsiyalar: {kmeans.n_iter_}\n"
            f"   • Inertia: {kmeans.inertia_:.2f}"
        )

        # Yuborish
        await query.message.reply_photo(
            photo=img,
            caption=info_text,
            parse_mode='HTML'
        )

        # Bazaga saqlash
        db.add_analysis(
            user_id=update.effective_user.id,
            algorithm='K-Means',
            dataset_name=context.user_data.get('dataset_name'),
            parameters={'k': k},
            n_clusters=k
        )

        await query.message.reply_text(
            "✅ <b>Tahlil tugadi!</b>\n\n"
            "Yangi tahlil uchun /analyze",
            parse_mode='HTML'
        )

        return ConversationHandler.END

    async def dbscan_eps_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Epsilon tanlandi"""
        query = update.callback_query
        await query.answer()

        if query.data == 'eps_custom':
            await query.edit_message_text(
                "✏️ Epsilon qiymatini yozing (0.1 - 2.0):\n\n"
                "Bekor qilish: /cancel"
            )
            context.user_data['waiting_custom_eps'] = True
            return DBSCAN_EPS

        eps = float(query.data.split('_')[1])
        context.user_data['eps'] = eps

        # MinPts tanlash
        keyboard = []
        for minpts in [3, 5, 7, 10, 15]:
            keyboard.append([InlineKeyboardButton(
                f"MinPts = {minpts}",
                callback_data=f'minpts_{minpts}'
            )])
        keyboard.append([InlineKeyboardButton("✏️ Boshqa qiymat", callback_data='minpts_custom')])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"✅ Epsilon: {eps}\n\n"
            "🔢 <b>MinPts qiymatini tanlang:</b>\n\n"
            "Bu Core Point bo'lish uchun kerakli minimum qo'shnilar soni.",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return DBSCAN_MINPTS

    async def dbscan_custom_eps(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Custom epsilon"""
        if not context.user_data.get('waiting_custom_eps'):
            return

        try:
            eps = float(update.message.text)

            if eps < 0.1 or eps > 2.0:
                await update.message.reply_text("❌ Qiymat 0.1 - 2.0 oralig'ida bo'lishi kerak!")
                return DBSCAN_EPS

            context.user_data['eps'] = eps
            context.user_data['waiting_custom_eps'] = False

            # MinPts tanlash
            keyboard = []
            for minpts in [3, 5, 7, 10, 15]:
                keyboard.append([InlineKeyboardButton(
                    f"MinPts = {minpts}",
                    callback_data=f'minpts_{minpts}'
                )])
            keyboard.append([InlineKeyboardButton("✏️ Boshqa qiymat", callback_data='minpts_custom')])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"✅ Epsilon: {eps}\n\n"
                "🔢 <b>MinPts qiymatini tanlang:</b>\n\n"
                "Bu Core Point bo'lish uchun kerakli minimum qo'shnilar soni.",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )

            return DBSCAN_MINPTS

        except ValueError:
            await update.message.reply_text("❌ Noto'g'ri format! Raqam kiriting.")
            return DBSCAN_EPS

    async def dbscan_minpts_chosen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """MinPts tanlandi"""
        query = update.callback_query
        await query.answer()

        if query.data == 'minpts_custom':
            await query.edit_message_text(
                "✏️ MinPts qiymatini yozing (3 - 20):\n\n"
                "Bekor qilish: /cancel"
            )
            context.user_data['waiting_custom_minpts'] = True
            return DBSCAN_MINPTS

        minpts = int(query.data.split('_')[1])
        context.user_data['minpts'] = minpts

        # Tasdiqlash
        keyboard = [
            [InlineKeyboardButton("✅ Tahlilni Boshlash", callback_data='dbscan_confirm_yes')],
            [InlineKeyboardButton("🔙 Ortga", callback_data='dbscan_confirm_no')]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            f"📋 <b>Tahlil Sozlamalari:</b>\n\n"
            f"🎯 Algoritm: DBSCAN\n"
            f"📊 Dataset: {context.user_data.get('dataset_name')}\n"
            f"📏 Epsilon (ε): {context.user_data.get('eps')}\n"
            f"🔢 MinPts: {minpts}\n\n"
            "Davom etamizmi?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return DBSCAN_CONFIRM

    async def dbscan_custom_minpts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Custom MinPts"""
        if not context.user_data.get('waiting_custom_minpts'):
            return

        try:
            minpts = int(update.message.text)

            if minpts < 3 or minpts > 20:
                await update.message.reply_text("❌ Qiymat 3 - 20 oralig'ida bo'lishi kerak!")
                return DBSCAN_MINPTS

            context.user_data['minpts'] = minpts
            context.user_data['waiting_custom_minpts'] = False

            # Tasdiqlash
            keyboard = [
                [InlineKeyboardButton("✅ Tahlilni Boshlash", callback_data='dbscan_confirm_yes')],
                [InlineKeyboardButton("🔙 Ortga", callback_data='dbscan_confirm_no')]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"📋 <b>Tahlil Sozlamalari:</b>\n\n"
                f"🎯 Algoritm: DBSCAN\n"
                f"📊 Dataset: {context.user_data.get('dataset_name')}\n"
                f"📏 Epsilon (ε): {context.user_data.get('eps')}\n"
                f"🔢 MinPts: {minpts}\n\n"
                "Davom etamizmi?",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )

            return DBSCAN_CONFIRM

        except ValueError:
            await update.message.reply_text("❌ Noto'g'ri format! Butun son kiriting.")
            return DBSCAN_MINPTS

    async def dbscan_confirmed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """DBSCAN tahlilni boshlash"""
        query = update.callback_query
        await query.answer()

        if query.data == 'dbscan_confirm_no':
            await query.edit_message_text("❌ Bekor qilindi.")
            return ConversationHandler.END

        await query.edit_message_text("⏳ <b>Tahlil boshlanmoqda...</b>", parse_mode='HTML')

        # Ma'lumotlarni olish
        X = context.user_data.get('data')
        eps = context.user_data.get('eps')
        minpts = context.user_data.get('minpts')

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_pts=minpts)
        dbscan.fit(X)

        # Grafik
        img = self.viz.plot_dbscan(X, dbscan, f"DBSCAN (ε={eps}, MinPts={minpts})")

        # Klaster ma'lumotlari
        cluster_info = dbscan.get_cluster_info()
        info_text = "📊 <b>Klaster Ma'lumotlari:</b>\n\n"

        for cluster in cluster_info:
            info_text += (
                f"🔹 <b>Klaster {cluster['cluster_id']}</b>\n"
                f"   • Nuqtalar: {cluster['n_points']}\n"
                f"   • Foiz: {cluster['percentage']:.1f}%\n\n"
            )

        info_text += (
            f"📈 <b>Umumiy:</b>\n"
            f"   • Topilgan klasterlar: {dbscan.n_clusters_}\n"
            f"   • Shovqin nuqtalari: {dbscan.n_noise_}\n"
            f"   • Core Points: {len(dbscan.core_points)}"
        )

        # Yuborish
        await query.message.reply_photo(
            photo=img,
            caption=info_text,
            parse_mode='HTML'
        )

        # Bazaga saqlash
        db.add_analysis(
            user_id=update.effective_user.id,
            algorithm='DBSCAN',
            dataset_name=context.user_data.get('dataset_name'),
            parameters={'eps': eps, 'minpts': minpts},
            n_clusters=dbscan.n_clusters_,
            n_noise_points=dbscan.n_noise_
        )

        await query.message.reply_text(
            "✅ <b>Tahlil tugadi!</b>\n\n"
            "Yangi tahlil uchun /analyze",
            parse_mode='HTML'
        )

        return ConversationHandler.END

    async def run_comparison(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Algoritmlarni taqqoslash"""
        if update.callback_query:
            msg = update.callback_query.message
            await update.callback_query.answer()
        else:
            msg = update.message

        await msg.reply_text("⏳ <b>Taqqoslash boshlanmoqda...</b>", parse_mode='HTML')

        X = context.user_data.get('data')

        # K-Means
        kmeans = KMeans(k=3, random_state=42)
        kmeans.fit(X)

        # DBSCAN
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        dbscan.fit(X)

        # Taqqoslash grafigi
        img = self.viz.plot_comparison(X, kmeans, dbscan)

        comparison_text = (
            "⚖️ <b>Algoritmlar Taqqoslash</b>\n\n"
            "<b>K-Means:</b>\n"
            f"   • Klasterlar: {kmeans.k}\n"
            f"   • Inertia: {kmeans.inertia_:.2f}\n"
            f"   • Iteratsiyalar: {kmeans.n_iter_}\n\n"
            "<b>DBSCAN:</b>\n"
            f"   • Klasterlar: {dbscan.n_clusters_}\n"
            f"   • Shovqin: {dbscan.n_noise_}\n"
            f"   • Core Points: {len(dbscan.core_points)}\n\n"
            "💡 <b>Xulosa:</b>\n"
            "K-Means dumaloq klasterlar uchun, DBSCAN murakkab shakllar uchun yaxshi!"
        )

        await msg.reply_photo(
            photo=img,
            caption=comparison_text,
            parse_mode='HTML'
        )

        # Bazaga saqlash
        db.add_analysis(
            user_id=update.effective_user.id,
            algorithm='Comparison',
            dataset_name=context.user_data.get('dataset_name'),
            parameters={'kmeans_k': 3, 'dbscan_eps': 0.3},
            n_clusters=kmeans.k
        )

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Tahlillar tarixi"""
        user_id = update.effective_user.id
        history = db.get_user_history(user_id, limit=10)

        if not history:
            await update.message.reply_text(
                "📭 <b>Tarix bo'sh!</b>\n\n"
                "Birinchi tahlil uchun /analyze",
                parse_mode='HTML'
            )
            return

        text = "📜 <b>Oxirgi 10 ta Tahlil:</b>\n\n"

        for i, (algo, dataset, n_clusters, date) in enumerate(history, 1):
            text += (
                f"{i}. <b>{algo}</b>\n"
                f"   📊 Dataset: {dataset}\n"
                f"   🔢 Klasterlar: {n_clusters}\n"
                f"   📅 Sana: {date}\n\n"
            )

        await update.message.reply_text(text, parse_mode='HTML')

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Statistika"""
        user_id = update.effective_user.id
        stats = db.get_user_stats(user_id)

        if not stats:
            await update.message.reply_text("❌ Ma'lumot topilmadi.")
            return

        total_analyses, join_date = stats

        text = (
            f"📊 <b>Sizning Statistikangiz</b>\n\n"
            f"👤 Foydalanuvchi: {update.effective_user.first_name}\n"
            f"📅 Qo'shilgan sana: {join_date}\n"
            f"🔢 Jami tahlillar: {total_analyses}\n\n"
            f"Davom eting! 🚀"
        )

        await update.message.reply_text(text, parse_mode='HTML')

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bekor qilish"""
        if update.message:
            await update.message.reply_text("❌ Bekor qilindi.")
        else:
            await update.callback_query.message.reply_text("❌ Bekor qilindi.")

        return ConversationHandler.END

    async def send_typing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Typing action"""
        if update.callback_query:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action='typing'
            )
        else:
            await update.message.chat.send_action('typing')


def main():
    """Botni ishga tushirish"""

    # Bot instance
    bot = ClusteringBot()

    # Application
    app = Application.builder().token(config.BOT_TOKEN).build()

    # Conversation Handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('analyze', bot.analyze)],
        states={
            CHOOSING_ALGORITHM: [CallbackQueryHandler(bot.algorithm_chosen, pattern='^algo_')],
            CHOOSING_SOURCE: [CallbackQueryHandler(bot.source_chosen, pattern='^source_')],
            CHOOSING_DATASET: [CallbackQueryHandler(bot.dataset_chosen, pattern='^dataset_')],
            UPLOADING_FILE: [
                MessageHandler(filters.Document.ALL, bot.file_uploaded),
                CommandHandler('cancel', bot.cancel)
            ],
            KMEANS_K: [CallbackQueryHandler(bot.kmeans_k_chosen, pattern='^k_')],
            KMEANS_CONFIRM: [CallbackQueryHandler(bot.kmeans_confirmed, pattern='^confirm_')],
            DBSCAN_EPS: [
                CallbackQueryHandler(bot.dbscan_eps_chosen, pattern='^eps_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.dbscan_custom_eps)
            ],
            DBSCAN_MINPTS: [
                CallbackQueryHandler(bot.dbscan_minpts_chosen, pattern='^minpts_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bot.dbscan_custom_minpts)
            ],
            DBSCAN_CONFIRM: [CallbackQueryHandler(bot.dbscan_confirmed, pattern='^dbscan_confirm_')],
        },
        fallbacks=[
            CommandHandler('cancel', bot.cancel),
            CallbackQueryHandler(bot.cancel, pattern='^cancel$')
        ]
    )

    # Handlers
    app.add_handler(CommandHandler('start', bot.start))
    app.add_handler(CommandHandler('help', bot.help_command))
    app.add_handler(CommandHandler('about', bot.about_command))
    app.add_handler(CommandHandler('history', bot.history))
    app.add_handler(CommandHandler('stats', bot.stats))
    app.add_handler(conv_handler)

    # Botni ishga tushirish
    logger.info("🤖 Bot ishga tushdi!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()