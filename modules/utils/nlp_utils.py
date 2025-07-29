import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )

    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def extract_skills(self, text):
        """Extract potential skills from text."""
        doc = nlp(text)
        skills = set()
        
        # Common skill indicators
        skill_indicators = ['experience with', 'proficient in', 'skilled in', 
                          'expertise in', 'knowledge of', 'familiar with']
        
        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            if any(indicator in chunk.text.lower() for indicator in skill_indicators):
                skills.add(chunk.text.lower())
        
        # Extract named entities that might be technologies or tools
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'GPE']:
                skills.add(ent.text.lower())
        
        return list(skills)

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using TF-IDF and cosine similarity."""
        # Preprocess both texts
        text1_processed = self.preprocess_text(text1)
        text2_processed = self.preprocess_text(text2)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([text1_processed, text2_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)

    def extract_keywords(self, text, n=10):
        """Extract top keywords from text."""
        doc = nlp(text)
        
        # Get word frequencies
        word_freq = {}
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                word_freq[token.text.lower()] = word_freq.get(token.text.lower(), 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:n]]

    def analyze_sentiment(self, text):
        """Analyze sentiment of text."""
        doc = nlp(text)
        
        # Simple sentiment analysis based on positive/negative word lists
        positive_words = {'good', 'great', 'excellent', 'positive', 'strong', 
                         'impressive', 'outstanding', 'successful', 'effective'}
        negative_words = {'poor', 'weak', 'negative', 'unsuccessful', 'ineffective',
                         'difficult', 'challenging', 'problematic'}
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5  # Neutral if no sentiment words found
        
        return positive_count / total

    def extract_education(self, text):
        """Extract education information from text."""
        doc = nlp(text)
        education = []
        
        # Common education degree patterns
        degree_patterns = [
            r'\b(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.E\.|M\.E\.|B\.Tech|M\.Tech)\b',
            r'\b(?:University|College|Institute|School)\b'
        ]
        
        # Find sentences containing education-related terms
        for sent in doc.sents:
            if any(re.search(pattern, sent.text, re.IGNORECASE) for pattern in degree_patterns):
                education.append(sent.text.strip())
        
        return education

    def extract_experience(self, text):
        """Extract work experience information from text."""
        doc = nlp(text)
        experience = []
        
        # Common experience indicators
        experience_indicators = [
            'experience', 'worked', 'employed', 'position', 'role',
            'responsibilities', 'duties', 'achievements'
        ]
        
        # Find sentences containing experience-related terms
        for sent in doc.sents:
            if any(indicator in sent.text.lower() for indicator in experience_indicators):
                experience.append(sent.text.strip())
        
        return experience 