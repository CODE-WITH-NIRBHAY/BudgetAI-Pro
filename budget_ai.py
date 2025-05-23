import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
import pyttsx3
import re
import random
import os
from datetime import datetime
from warnings import warn

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
    def listen(self):
        with self.mic as source:
            print("\n🎤 Speak now (e.g., '500 rupees for pizza')...")
            audio = self.recognizer.listen(source, phrase_time_limit=5)
        try:
            return self.recognizer.recognize_google(audio).lower()
        except:
            return None
            
    def speak(self, text):
        print(f"🔊 AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

class ExpenseClassifier:
    def __init__(self):
        self.patterns = {
            "Food": r"(zomato|swiggy|food|pizza|burger|coffee|lunch|dinner|groceries)",
            "Transport": r"(uber|ola|taxi|bus|train|metro|petrol|fuel)",
            "Rent": r"(rent|room|pg|hostel|deposit|lease)"
        }
        
    def classify(self, text):
        text = text.lower()
        for category, pattern in self.patterns.items():
            if re.search(pattern, text):
                return category
        return "Others"

class BudgetAI:
    def __init__(self, data_file="expenses.csv"):
        self.file = data_file
        self.voice = VoiceAssistant()
        self.classifier = ExpenseClassifier()
        self.df = self._load_data()
        
        self.tips_db = {
            "Food": [
                "🍱 Meal prepping can save you ₹2000/month!",
                "☕ Making coffee at home saves ₹150/week!"
            ],
            "Transport": [
                "🚲 Cycling to college 2x/week = ₹800 saved!",
                "🚗 Carpool with 3 friends to split fuel costs!"
            ],
            "Rent": [
                "🏠 Negotiate rent during summer vacations!",
                "💡 Switch to LED bulbs to reduce electricity bills!"
            ]
        }
        
    def _load_data(self):
        try:
            df = pd.read_csv(
                self.file, 
                parse_dates=['Date'],
                dtype={'Amount': 'float32', 'Category': 'category'}
            )
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"⚠️ Created new file: {str(e)}")
            return pd.DataFrame(columns=["Amount", "Category", "Date"])

    def _optimize_dataframe(self):
        if not self.df.empty:
            self.df['Category'] = self.df['Category'].astype('category')
            self.df['Amount'] = pd.to_numeric(self.df['Amount'], downcast='float')

    def log_expense(self, amount, description, date=None):
        try:
            category = self.classifier.classify(description)
            new_entry = {
                "Amount": float(amount),
                "Category": category,
                "Date": pd.to_datetime(date) if date else datetime.now()
            }
            self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
            self._optimize_dataframe()
            self.df.to_csv(self.file, index=False)
            return True
        except Exception as e:
            print(f"❌ Error logging expense: {str(e)}")
            return False
        
    def voice_log_expense(self):
        try:
            spoken = self.voice.listen()
            if not spoken:
                raise ValueError("Couldn't understand audio")
                
            amount = re.search(r"(\d+)\s*(rs|rupees|₹)?", spoken)
            desc = re.sub(r"\d+\s*(rs|rupees|₹)?\s*", "", spoken).strip()
            
            if not amount:
                raise ValueError("No amount detected")
                
            success = self.log_expense(amount.group(1), desc)
            if success:
                self.voice.speak(f"Logged ₹{amount.group(1)} for {desc}")
            return success
            
        except Exception as e:
            self.voice.speak(f"Error: {str(e)}")
            return False

    def get_analysis(self):
        if self.df.empty:
            return {"error": "No expenses logged yet"}
            
        try:
            analysis = {
                "total": round(self.df['Amount'].sum(), 2),
                "top_category": self.df['Category'].mode()[0],
                "monthly": self.df.groupby(
                    self.df['Date'].dt.to_period('M'))['Amount'].sum().to_dict()
            }
            return analysis
        except Exception as e:
            return {"error": f"Analysis error: {str(e)}"}

    def get_ai_tip(self):
        if self.df.empty:
            return "💡 Start by logging your first expense!"
            
        try:
            top_cat = self.df['Category'].mode()[0]
            return random.choice(self.tips_db.get(top_cat, [
                "💰 Save 10% of every paycheck automatically!"
            ]))
        except:
            return "💡 Track your expenses to get personalized tips!"

    def predict_spending(self):
        if len(self.df) < 7:
            return "Need at least 7 entries for accurate predictions"
            
        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Days'] = (self.df['Date'] - self.df['Date'].min()).dt.days
            
            model = LinearRegression()
            model.fit(self.df[['Days']], self.df['Amount'])
            
            next_day = self.df['Days'].max() + 1
            prediction = model.predict([[next_day]])[0]
            
            trend = "↑ Increasing" if model.coef_[0] > 0 else "↓ Decreasing"
            return f"Predicted: ₹{prediction:.2f} ({trend} trend)"
        
        except Exception as e:
            return f"Prediction error: {str(e)}"

    def show_insights(self):
        if self.df.empty:
            print("No data to visualize")
            return
            
        try:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum()
            monthly.plot(kind='bar', color='#4CAF50')
            plt.title("Monthly Spending Trend")
            plt.ylabel("Amount (₹)")
            
            plt.subplot(1, 3, 2)
            self.df['Category'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                colors=['#FF5722', '#2196F3', '#9C27B0', '#607D8B']
            )
            plt.title("Spending by Category")
            
            plt.subplot(1, 3, 3)
            self.df['Weekday'] = self.df['Date'].dt.day_name()
            self.df['Hour'] = self.df['Date'].dt.hour
            pd.pivot_table(
                self.df, 
                values='Amount', 
                index='Weekday', 
                columns='Hour',
                aggfunc='sum',
                fill_value=0
            ).plot(kind='box', vert=False)
            plt.title("Spending Patterns")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Chart error: {str(e)}")

def run_app():
    ai = BudgetAI()
    
    while True:
        print("\n" + "="*40)
        print("💰 AI BUDGET TRACKER PRO".center(40))
        print("="*40)
        print("1. Log Expense (Typing)")
        print("2. Log Expense (Voice)")
        print("3. View Analysis")
        print("4. Get AI Tip")
        print("5. Predict Spending")
        print("6. Show Insights")
        print("7. Exit")
        
        choice = input("\nChoose (1-7): ").strip()
        
        if choice == "1":
            try:
                amount = float(input("Amount (₹): "))
                desc = input("Description: ")
                if ai.log_expense(amount, desc):
                    print("✅ Expense logged!")
            except ValueError:
                print("❌ Invalid input!")
                
        elif choice == "2":
            print("\nSpeak naturally like: '300 rupees for dominos pizza'")
            if ai.voice_log_expense():
                print("✅ Voice expense logged!")
                
        elif choice == "3":
            analysis = ai.get_analysis()
            if "error" in analysis:
                print(analysis["error"])
            else:
                print(f"\n📊 Total Spent: ₹{analysis['total']:.2f}")
                print(f"🏆 Top Category: {analysis['top_category']}")
                print("📅 Monthly Breakdown:")
                for month, amount in analysis['monthly'].items():
                    print(f"  - {month}: ₹{amount:.2f}")
                
        elif choice == "4":
            print("\n" + ai.get_ai_tip())
            
        elif choice == "5":
            print("\n" + ai.predict_spending())
            
        elif choice == "6":
            ai.show_insights()
            
        elif choice == "7":
            ai.voice.speak("Goodbye! Keep saving smart!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    run_app()