import streamlit as st
import whisper
import os
import tempfile
import sys
import uuid
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import re

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from topic_segmentation import TopicSegmenter
from audio_preprocessor import AudioPreprocessor

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="EchoAI - Automated Podcast Transcription & Insights",
    layout="wide"
)

# -------------------- LOAD RESOURCES --------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")  # Fast & CPU-safe

@st.cache_resource
def load_modules():
    return TopicSegmenter(), AudioPreprocessor()

whisper_model = load_whisper()
segmenter, preprocessor = load_modules()

# -------------------- FUNCTIONS --------------------
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    # Return both full text and segments for timestamps
    return {
        "text": result["text"],
        "segments": result["segments"]
    }

def detect_song_mode(text, raw_segments):
    """
    Detects if the audio is likely a song based on:
    - High repetition of lines (chorus pattern)
    - Short segment duration patterns
    - High density of similar phrases
    """
    if not raw_segments or len(raw_segments) < 3:
        return False
    
    # Check for high repetition patterns
    lines = [seg["text"].strip().lower() for seg in raw_segments if seg["text"].strip()]
    if len(lines) < 5:
        return False
    
    # Count repeated lines (exact or near-exact matches)
    unique_lines = set(lines)
    repetition_ratio = 1 - (len(unique_lines) / len(lines))
    
    # If more than 30% repetition, likely a song
    return repetition_ratio > 0.30

def clean_song_lyrics(text, raw_segments):
    """
    Removes duplicate/repeated lines (chorus) from song lyrics.
    Preserves unique lines while maintaining narrative flow.
    """
    lines = []
    seen_lines = set()
    
    for seg in raw_segments:
        line = seg["text"].strip()
        if not line:
            continue
        
        # Normalize for comparison
        line_normalized = re.sub(r'[^\w\s]', '', line.lower())
        line_normalized = re.sub(r'\s+', ' ', line_normalized).strip()
        
        # Skip if we've seen this line before (chorus detection)
        if line_normalized in seen_lines and len(line_normalized) > 10:
            continue
        
        seen_lines.add(line_normalized)
        lines.append(line)
    
    return " ".join(lines)

def time_based_segmentation(raw_segments, num_segments=4):
    """
    Divides transcript into equal-duration chunks based on Whisper timestamps.
    """
    if not raw_segments:
        return []
    
    total_duration = raw_segments[-1]["end"]
    segment_duration = total_duration / num_segments
    
    segments = []
    current_segment_text = []
    current_start = 0
    target_end = segment_duration
    segment_idx = 0
    
    for seg in raw_segments:
        seg_end = seg["end"]
        
        # Add text to current segment
        current_segment_text.append(seg["text"].strip())
        
        # Check if we've reached the target end time for this segment
        if seg_end >= target_end and segment_idx < num_segments - 1:
            segments.append({
                "text": " ".join(current_segment_text),
                "start_time": current_start,
                "end_time": seg_end
            })
            segment_idx += 1
            current_start = seg_end
            target_end = (segment_idx + 1) * segment_duration
            current_segment_text = []
    
    # Add remaining text to final segment
    if current_segment_text:
        segments.append({
            "text": " ".join(current_segment_text),
            "start_time": current_start,
            "end_time": total_duration
        })
    
    return segments

def generate_song_summary(text):
    """
    Generates a summary focused on emotional theme, message, and narrative for songs.
    Incorporates techniques from advanced LLM analysis for better quality.
    """
    # Split text into sentences for better context analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Extract key emotional and thematic words
    emotional_keywords = [
        "love", "heart", "pain", "joy", "dream", "hope", "fear", "freedom",
        "home", "alone", "together", "lost", "found", "light", "dark",
        "forever", "never", "always", "remember", "forget", "believe",
        "longing", "yearning", "passion", "sorrow", "bliss", "anguish",
        "desire", "regret", "trust", "betrayal", "healing", "wounded"
    ]
    
    # Extract story progression indicators
    progression_keywords = [
        "once", "then", "later", "after", "before", "when", "while", "until",
        "beginning", "middle", "end", "first", "next", "finally", "eventually"
    ]
    
    text_lower = text.lower()
    found_emotions = [kw for kw in emotional_keywords if kw in text_lower]
    found_progression = [kw for kw in progression_keywords if kw in text_lower]
    
    # Identify emotional tone
    positive_words = ["love", "joy", "happy", "light", "hope", "dream", "together", "bliss"]
    negative_words = ["pain", "sad", "dark", "alone", "lost", "fear", "cry", "tears", "sorrow"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    emotional_tone = ""
    if pos_count > neg_count:
        emotional_tone = "uplifting"
    elif neg_count > pos_count:
        emotional_tone = "melancholic"
    else:
        emotional_tone = "reflective"
    
    # Get beginning and end for narrative context
    first_sentence = sentences[0] if sentences else ""
    last_sentence = sentences[-1] if sentences else ""
    
    # Build a more nuanced summary with emotional theme and story progression
    summary_parts = []
    
    # Opening with emotional context
    if first_sentence:
        summary_parts.append(f"This {emotional_tone} song begins with {first_sentence.lower()}.")
    
    # Central theme based on emotions
    if found_emotions:
        main_themes = found_emotions[:4]  # Take top 4 themes
        if len(main_themes) > 1:
            themes_str = ", ".join(main_themes[:-1]) + f" and {main_themes[-1]}"
        else:
            themes_str = main_themes[0] if main_themes else "emotional themes"
        summary_parts.append(f"The lyrics explore themes of {themes_str}.")
    
    # Story progression if present
    if found_progression and len(found_progression) > 1:
        summary_parts.append(f"The narrative unfolds through {len(found_progression)} distinct stages.")
    
    # Closing with resolution
    if last_sentence and last_sentence != first_sentence:
        summary_parts.append(f"It concludes with {last_sentence.lower()}.")
    
    # Final fallback
    if not summary_parts:
        summary_parts.append("A musical piece with lyrical content.")
    
    # Join and ensure maximum 3 sentences
    full_summary = " ".join(summary_parts)
    summary_sentences = re.split(r'[.!?]+', full_summary)
    summary_sentences = [s.strip() for s in summary_sentences if s.strip()][:3]  # Max 3 sentences
    
    return " ".join([s + "." for s in summary_sentences])

def generate_segment_summary_song(text):
    """
    Generates a short summary for a song segment focusing on emotional progression.
    """
    # Extract key phrases (first few meaningful words)
    words = text.split()[:8]
    opening = " ".join(words)
    
    # Detect emotional tone
    positive_words = ["love", "joy", "happy", "light", "hope", "dream", "together"]
    negative_words = ["pain", "sad", "dark", "alone", "lost", "fear", "cry", "tears"]
    
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    tone = ""
    if pos_count > neg_count:
        tone = "uplifting"
    elif neg_count > pos_count:
        tone = "melancholic"
    else:
        tone = "reflective"
    
    return f"{opening}... ({tone} tone)"

def process_segments(transcript_data, algorithm="Similarity", is_song=False):
    text = transcript_data["text"]
    raw_segments = transcript_data["segments"]
    
    # Detect if this is a song
    detected_song = detect_song_mode(text, raw_segments)
    is_song = is_song or detected_song
    
    if is_song:
        # Clean lyrics by removing chorus repetitions
        cleaned_lyrics = clean_song_lyrics(text, raw_segments)
        
        # Use time-based segmentation for songs (dynamic segments based on duration)
        total_duration = raw_segments[-1]["end"] if raw_segments else 0.0
        # Calculate number of segments based on duration: ~1 segment per minute, minimum 3, maximum 8
        duration_minutes = total_duration / 60
        num_segments = max(3, min(8, int(duration_minutes + 1)))  # 1-3 min -> 2-4 segments, 4-5 min -> 5-6 segments, etc.
        time_segments = time_based_segmentation(raw_segments, num_segments)
        
        # Generate overall song summary
        overall_summary = generate_song_summary(cleaned_lyrics)
        
        # Process each time-based segment
        processed = []
        for i, seg in enumerate(time_segments):
            content = seg["text"]
            start_time = seg["start_time"]
            end_time = seg["end_time"]
            
            # Clean content
            content_clean = clean_text(content)
            keywords = segmenter.extract_keywords(content_clean)
            segment_summary = generate_segment_summary_song(content_clean)
            sentiment_label, sentiment_score, sentiment_color = analyze_sentiment(content_clean)
            topic_title = segmenter.generate_title(content_clean, keywords)
            
            processed.append({
                "id": i,
                "label": f"Topic {i+1}: {topic_title}",
                "title": topic_title,
                "text": content_clean,
                "keywords": keywords,
                "summary": segment_summary,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "sentiment_color": sentiment_color,
                "start_time": start_time,
                "end_time": end_time,
                "is_song_segment": True
            })
        
        # Store overall summary in session for display
        st.session_state.song_overall_summary = overall_summary
        st.session_state.is_song_analysis = True
        return processed
    
    # Standard podcast processing (existing logic)
    if algorithm == "Similarity (Fast)":
        segmented_texts = segmenter.segment_with_similarity(text)
    elif algorithm == "TextTiling (NLTK)":
        segmented_texts = segmenter.segment_with_texttiling(text)
    elif algorithm == "Embeddings (Advanced)":
        segmented_texts = segmenter.segment_with_embeddings(text)
    else:
        segmented_texts = segmenter.segment_with_similarity(text)
    
    # Enforce topic count adaptively based on audio length
    total_audio_duration = raw_segments[-1]["end"] if raw_segments else 0.0
    segmented_texts = segmenter.enforce_topic_count(segmented_texts, duration=total_audio_duration)
    
    # Map segmented texts back to timestamps
    processed = []
    current_raw_idx = 0
    
    for i, seg in enumerate(segmented_texts):
        content = seg["text"]
        
        # Determine start/end times
        start_time = None
        end_time = None
        
        if current_raw_idx < len(raw_segments):
            start_time = raw_segments[current_raw_idx]["start"]
            accumulated_text = ""
            while current_raw_idx < len(raw_segments):
                seg_text = raw_segments[current_raw_idx]["text"].strip()
                accumulated_text += " " + seg_text
                end_time = raw_segments[current_raw_idx]["end"]
                current_raw_idx += 1
                # Increase match sensitivity and ensure we don't skip too much
                if len(accumulated_text) >= len(content) * 0.85:
                    break
        
        # Ensure the last segment covers the end of the audio
        if i == len(segmented_texts) - 1:
            end_time = total_audio_duration

        # New Week 5 Analysis
        content = clean_text(content)
        keywords = segmenter.extract_keywords(content)
        raw_summary = segmenter.summarize(content)
        polished = polish_summary(raw_summary)
        sentiment_label, sentiment_score, sentiment_color = analyze_sentiment(content)
        
        # Generate context-aware title
        topic_title = segmenter.generate_title(content, keywords)
        
        processed.append({
            "id": i,
            "label": f"Topic {i+1}: {topic_title}",
            "title": topic_title,
            "text": content,
            "keywords": keywords,
            "summary": polished,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "sentiment_color": sentiment_color,
            "start_time": start_time or 0.0,
            "end_time": end_time or 0.0
        })
    return processed

def generate_timeline(segments, selected_id=None):
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Calculate total duration
    if not segments:
        return fig
        
    total_duration = segments[-1]["end_time"]
    
    # Use a colormap
    cmap = plt.get_cmap("tab20")
    
    for i, seg in enumerate(segments):
        start = seg["start_time"]
        width = seg["end_time"] - start
        
        # Color: Highlight selected
        color = cmap(i % 20)
        alpha = 1.0 if (selected_id is None or seg["id"] == selected_id) else 0.3
        
        ax.barh(0, width, left=start, height=0.5, color=color, alpha=alpha, edgecolor='white')
        
        # Add label if space permits
        if width > total_duration * 0.05:
            ax.text(start + width/2, 0, f"T{i+1}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

def format_timestamp(seconds):
    """
    Converts seconds to MM:SS format with leading zeros.
    """
    if seconds is None:
        return "UNKNOWN"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def clean_text(text):
    """
    Cleans raw text to remove numeric artifacts and Whisper noise.
    """
    # Remove Whisper-specific noise artifacts like [Music], [Applause], or [00:00.000]
    cleaned = re.sub(r'\[.*?\]', '', text)
    # Remove repeated/stray Whisper artifacts (often short numbers like "2000 2000" or single digits at start)
    cleaned = re.sub(r'\b\d{1,4}\b\s+\b\d{1,4}\b', '', cleaned)
    cleaned = re.sub(r'^\d+\s+', '', cleaned, flags=re.MULTILINE)
    # General whitespace cleanup
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1.0 to 1.0
    
    # Scale -1.0..1.0 to 1..10
    score = round((polarity + 1) * 4.5 + 1, 1)
    
    if score <= 3.0:
        label = "NEGATIVE"
        color = "red"
    elif score <= 7.0:
        label = "NEUTRAL"
        color = "orange"
    else:
        label = "POSITIVE"
        color = "green"
        
    return label, score, color

def generate_wordcloud(keywords):
    if not keywords:
        return None
    # Use frequencies if keywords are just a list
    word_freq = {word: len(keywords) - i for i, word in enumerate(keywords)}
    wc = WordCloud(background_color="white", width=400, height=200).generate_from_frequencies(word_freq)
    return wc.to_array()

def polish_summary(summary):
    # Remove fillers
    fillers = [r"\buh\b", r"\bum\b", r"\byou know\b", r"\blike\b"]
    cleaned = summary
    for f in fillers:
        cleaned = re.sub(f, "", cleaned, flags=re.IGNORECASE)
    
    # Fix spacing and capitalization
    cleaned = cleaned.strip().capitalize()
    if not cleaned.endswith("."):
        cleaned += "."
        
    # Limit to 2-3 sentences (sent_tokenize is already in segmenter)
    sentences = segmenter.summarize(cleaned, num_sentences=3)
    return sentences

# -------------------- UI LAYOUT --------------------
st.title("🎙️ EchoAI - Automated Podcast Transcription & Insights")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    algo_choice = st.selectbox(
        "Segmentation Algorithm",
        ["Embeddings (Advanced)", "TextTiling (NLTK)", "Similarity (Fast)"],
        index=0,
        help="Embeddings use AI (Sentence-BERT) to understand the meaning of the text for smart segmentation."
    )
    

    st.info("System automatically preprocesses audio (Denoise & Normalize) before transcription.")

# State Management
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'segments' not in st.session_state:
    st.session_state.segments = []

# 1. AUDIO INPUT
st.subheader("1️⃣ Audio Input")
uploaded_audio = st.file_uploader(
    "Upload audio file (MP3 / WAV / M4A)",
    type=["mp3", "wav", "m4a"],
    key="file_uploader"
)

if uploaded_audio is not None:
    # Check if it's a new file
    if uploaded_audio.name != st.session_state.last_uploaded_file:
        st.session_state.last_uploaded_file = uploaded_audio.name
        st.session_state.transcript = None
        st.session_state.segments = []
        # st.rerun() # Removed unnecessary rerun here to allow first interaction

    st.audio(uploaded_audio, format='audio/mp3')

    # Trigger Pipeline
    if st.button("🚀 Start Auto-Pipeline"):
        # Save original to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_audio.getvalue())
            raw_path = tmp.name
        
        processed_path = raw_path + "_processed.wav"

        try:
            # Step 1: Preprocessing
            with st.spinner("⚙️ Step 1: Preprocessing Audio (Denoising & Normalizing)..."):
                preprocessor.process(raw_path, processed_path)
                st.success("Preprocessing Done.")

            # Step 2: Transcription
            with st.spinner("📝 Step 2: Generating Full Transcription..."):
                raw_transcript_data = transcribe_audio(processed_path)
                # Apply text cleaning to the full transcript
                cleaned_full_text = clean_text(raw_transcript_data["text"])
                st.session_state.transcript = cleaned_full_text
                # Keep raw segments for segmentation logic
                st.session_state.raw_transcript_data = {
                    "text": cleaned_full_text,
                    "segments": raw_transcript_data["segments"]
                }
            
            # Step 3: Segmentation
            with st.spinner("📌 Step 3: Performing Topic Segmentation..."):
                # Detect if this is a song based on repetition patterns
                raw_segments = raw_transcript_data["segments"]
                is_song = detect_song_mode(cleaned_full_text, raw_segments)
                
                st.session_state.segments = process_segments(st.session_state.raw_transcript_data, algo_choice, is_song=is_song)
                
                # Set song flag in session state
                st.session_state.is_song_analysis = is_song
            
            st.success("Pipeline Completed Successfully!")
            
        except Exception as e:
            st.error(f"Error during processing: {e}")
        finally:
            # Cleanup
            if os.path.exists(raw_path):
                os.remove(raw_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)

# 2. FULL TRANSCRIPTION
if st.session_state.transcript:
    st.divider()
    
    # Check if this is a song analysis
    is_song_analysis = hasattr(st.session_state, 'is_song_analysis') and st.session_state.is_song_analysis
    
    if is_song_analysis:
        st.subheader("🎵 Song Analysis")
        
        # Display the overall song summary
        if hasattr(st.session_state, 'song_overall_summary') and st.session_state.song_overall_summary:
            st.markdown("**Full Summary:**")
            st.info(st.session_state.song_overall_summary)
        
        # Display topic segments
        st.markdown("**Topic Segments:**")
        
        # Display each topic segment in a structured format
        if st.session_state.segments:
            for segment in st.session_state.segments:
                with st.container():
                    st.markdown(f"**Topic {segment['id']+1}: {segment['title']}**")
                    start_fmt = format_timestamp(segment['start_time'])
                    end_fmt = format_timestamp(segment['end_time'])
                    st.markdown(f"*Time: {start_fmt} – {end_fmt}*", unsafe_allow_html=True)
                    st.success(f"Summary: {segment['summary']}")
                    
                    # Display sentiment
                    s_color = segment['sentiment_color']
                    st.markdown(f"<span style='color:{s_color}; font-weight:bold;'>{segment['sentiment_label']} (Score: {segment['sentiment_score']}/10)</span>", unsafe_allow_html=True)
                    
                    # Display keywords
                    if segment['keywords']:
                        kw_html = ""
                        for kw in segment['keywords']:
                            kw_html += f"<span style='background-color: #f0f2f6; color: #31333f; padding: 4px 12px; border-radius: 16px; margin: 4px; display: inline-block; border: 1px solid #dfe1e5; font-weight: 500;'>{kw}</span>"
                        st.markdown(
                            f"<div style='background-color: #f8f9fb; border: 1px solid #e6e9ef; border-radius: 8px; padding: 16px; margin-bottom: 20px;'>{kw_html}</div>",
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(f"**Transcript:**")
                    st.text_area("", segment["text"], height=100, key=f"song_txt_{segment['id']}", label_visibility="collapsed")
                    st.divider()
    
    st.subheader("2️⃣ Analysis Timeline")
    # Show interactive timeline
    selected_id = None
    if st.session_state.segments:
        # Find ID of selected radio option
        if 'selected_topic_label' in st.session_state:
            try:
                selected_id = next(s["id"] for s in st.session_state.segments if s["label"] == st.session_state.selected_topic_label)
            except StopIteration:
                pass
        
        fig = generate_timeline(st.session_state.segments, selected_id)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📝 Full Transcription")
    with st.expander("📄 View Complete Transcript", expanded=False):
        st.text_area("Raw Text", st.session_state.transcript, height=200)

# 3. SEGMENTATION
if st.session_state.segments:
    # Only show the detailed segmentation view if this is NOT a song analysis
    is_song_analysis = hasattr(st.session_state, 'is_song_analysis') and st.session_state.is_song_analysis
    
    if not is_song_analysis:
        st.divider()
        st.subheader("3️⃣ Topic Segmentation & Analysis")
        
        col_nav, col_content = st.columns([1, 3])
        
        with col_nav:
            st.markdown("**Topic List**")
            options = [s["label"] for s in st.session_state.segments]
            selected_label = st.radio("Select a Topic:", options, label_visibility="collapsed", key="selected_topic_label")
        
        selected_segment = next(s for s in st.session_state.segments if s["label"] == selected_label)
        
        with col_content:
            # 1. Topic Title & 2. Timestamp & Duration
            start_fmt = format_timestamp(selected_segment['start_time'])
            end_fmt = format_timestamp(selected_segment['end_time'])
            duration_sec = selected_segment['end_time'] - selected_segment['start_time']
            duration_fmt = format_timestamp(duration_sec)
            
            h_col1, h_col2 = st.columns([3, 1])
            with h_col1:
                st.markdown(f"#### Topic {selected_segment['id'] + 1}: {selected_segment['title']} ({start_fmt} – {end_fmt})")
                st.markdown(f"**Duration: {duration_fmt}**")
            with h_col2:
                st.markdown(f"<span style='color:{selected_segment['sentiment_color']}; font-weight:bold;'>{selected_segment['sentiment_label']} (Score: {selected_segment['sentiment_score']}/10)</span>", unsafe_allow_html=True)
                
            st.divider()
            
            # 3. Summary
            st.markdown("**Summary**")
            st.success(selected_segment['summary'])
                
            # 4. Keywords (Highlighted Box)
            st.markdown("**Keywords**")
            if selected_segment['keywords']:
                # Create a boxed container for keywords using custom CSS-like markdown
                kw_html = ""
                for kw in selected_segment['keywords']:
                    kw_html += f"<span style='background-color: #f0f2f6; color: #31333f; padding: 4px 12px; border-radius: 16px; margin: 4px; display: inline-block; border: 1px solid #dfe1e5; font-weight: 500;'>{kw}</span>"
                
                st.markdown(
                    f"<div style='background-color: #f8f9fb; border: 1px solid #e6e9ef; border-radius: 8px; padding: 16px; margin-bottom: 20px;'>{kw_html}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No keywords identified for this segment.")

            # Optional: Word Cloud (Still valuable for visualization)
            with st.expander("☁️ View Topic Word Cloud", expanded=False):
                wc_img = generate_wordcloud(selected_segment['keywords'])
                if wc_img is not None:
                    st.image(wc_img, use_container_width=True)

            # 5. Transcript
            st.markdown("**Transcript**")
            st.text_area("Segment Text", selected_segment["text"], height=250, key=f"txt_{selected_segment['id']}", label_visibility="collapsed")
