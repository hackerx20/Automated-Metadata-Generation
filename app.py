import streamlit as st
import pandas as pd
import json
import time
import os
from datetime import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go




# Import custom modules
from src.document_processor import DocumentProcessor
from src.content_analyzer import ContentAnalyzer
from src.metadata_generator import MetadataGenerator
from src.utils import validate_file, format_file_size, get_file_hash

# Page configuration
st.set_page_config(
    page_title="Automated Metadata Generation System",
    page_icon="🎅",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MetadataApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.metadata_generator = MetadataGenerator()
        
        # Initialize session state
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'current_metadata' not in st.session_state:
            st.session_state.current_metadata = None
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None

    def main(self):
        """Main application interface"""
        st.title("🔍 Automated Metadata Generation System")
        st.markdown("Upload documents (PDF, DOCX, TXT) to automatically generate semantic metadata using advanced NLP techniques.")
        
        # Sidebar for navigation and history
        with st.sidebar:
            st.header("📊 System Overview")
            st.metric("Documents Processed", len(st.session_state.processing_history))
            
            if st.session_state.processing_history:
                st.subheader("📈 Processing History")
                for i, record in enumerate(reversed(st.session_state.processing_history[-5:])):
                    with st.expander(f"{record['filename'][:20]}..."):
                        st.write(f"**Size:** {format_file_size(record['file_size'])}")
                        st.write(f"**Type:** {record['file_type']}")
                        st.write(f"**Processed:** {record['timestamp']}")
                        st.write(f"**Processing Time:** {record['processing_time']:.2f}s")

        # Main interface tabs
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📋 Metadata Results", "📊 Analytics Dashboard"])
        
        with tab1:
            self.upload_interface()
        
        with tab2:
            self.display_metadata()
        
        with tab3:
            self.analytics_dashboard()

    def upload_interface(self):
        """File upload and processing interface"""
        st.header("📤 Document Upload")
        
        # File upload with error handling
        st.info("Supported formats: PDF, DOCX, TXT (Max size: 10MB)")
        
        try:
            uploaded_file = st.file_uploader(
                "Choose a document file",
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Upload your document for automated metadata generation",
                accept_multiple_files=False,
                key="document_uploader"
            )
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            st.info("Try refreshing the page or using a smaller file.")
            uploaded_file = None
        
        if uploaded_file is not None:
            try:
                # Show file info immediately
                st.success(f"File uploaded: {uploaded_file.name} ({format_file_size(uploaded_file.size)})")
                
                # File validation
                is_valid, error_message = validate_file(uploaded_file)
                
                if not is_valid:
                    st.error(f"File validation failed: {error_message}")
                    return
                
                # Display file information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Name", uploaded_file.name)
                with col2:
                    st.metric("File Size", format_file_size(uploaded_file.size))
                with col3:
                    file_type = uploaded_file.type if uploaded_file.type else "Unknown"
                    st.metric("File Type", file_type)
                
                # Processing button
                if st.button("Process Document", type="primary"):
                    with st.spinner("Processing document..."):
                        self.process_document(uploaded_file)
                        
            except Exception as e:
                st.error(f"Error handling file: {str(e)}")
                st.info("Please try uploading a different file or refresh the page.")

    def process_document(self, uploaded_file):
        """Process the uploaded document and generate metadata"""
        start_time = time.time()
        
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded file temporarily
            status_text.text("📁 Saving uploaded file...")
            progress_bar.progress(10)
            
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            temp_file_path = f"temp/{uploaded_file.name}"
            
            try:
                with open(temp_file_path, "wb") as f:
                    file_content = uploaded_file.getbuffer()
                    if len(file_content) == 0:
                        st.error("File appears to be empty")
                        return
                    f.write(file_content)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as write_error:
                st.error(f"Failed to save file: {str(write_error)}")
                return
            
            # Step 2: Extract text content
            status_text.text("📖 Extracting text content...")
            progress_bar.progress(25)
            
            extracted_text = self.doc_processor.extract_text(temp_file_path, uploaded_file.type)
            
            if not extracted_text.strip():
                st.error("No text content could be extracted from the document.")
                return
            
            # Extract PDF metadata if it's a PDF file
            pdf_metadata = None
            if uploaded_file.type == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
                try:
                    status_text.text("📋 Extracting PDF metadata...")
                    progress_bar.progress(35)
                    pdf_metadata = self.doc_processor.extract_pdf_metadata(temp_file_path)
                except Exception as e:
                    st.warning(f"Could not extract PDF metadata: {str(e)}")
            
            # Step 3: Preprocess text
            status_text.text("🔧 Preprocessing text...")
            progress_bar.progress(45)
            
            cleaned_text = self.doc_processor.preprocess_text(extracted_text)
            
            # Step 4: Analyze content
            status_text.text("🧠 Analyzing content with NLP...")
            progress_bar.progress(65)
            
            analysis_results = self.content_analyzer.analyze_content(cleaned_text)
            
            # Step 5: Generate metadata
            status_text.text("📊 Generating metadata...")
            progress_bar.progress(85)
            
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'upload_time': datetime.now().isoformat(),
                'content': uploaded_file.getbuffer()
            }
            
            metadata = self.metadata_generator.generate_metadata(
                file_info, cleaned_text, analysis_results, pdf_metadata
            )
            
            # Step 6: Finalize
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            processing_time = time.time() - start_time
            metadata['processing_info']['processing_time'] = processing_time
            
            # Store results in session state
            st.session_state.current_metadata = metadata
            st.session_state.current_document = {
                'filename': uploaded_file.name,
                'text': cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text
            }
            
            # Add to processing history
            history_record = {
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'file_type': uploaded_file.type,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': processing_time
            }
            st.session_state.processing_history.append(history_record)
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            st.success(f"✅ Document processed successfully in {processing_time:.2f} seconds!")
            st.balloons()
            
            # Auto-switch to results tab
            st.info("📋 Switch to the 'Metadata Results' tab to view the generated metadata.")
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            # Clean up temporary file on error
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def display_metadata(self):
        """Display generated metadata in organized tabs"""
        st.header("📋 Metadata Results")
        
        if st.session_state.current_metadata is None:
            st.info("📤 No metadata available. Please upload and process a document first.")
            return
        
        metadata = st.session_state.current_metadata
        document = st.session_state.current_document
        
        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            json_data = json.dumps(metadata, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Download JSON",
                json_data,
                file_name=f"{document['filename']}_metadata.json",
                mime="application/json"
            )
        
        with col2:
            # Create CSV from flattened metadata
            csv_data = self.metadata_to_csv(metadata)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name=f"{document['filename']}_metadata.csv",
                mime="text/csv"
            )
        
        # Metadata display tabs
        meta_tab1, meta_tab2, meta_tab3, meta_tab4, meta_tab5 = st.tabs([
            "📄 Overview", "🔍 Content Analysis", "🧠 Semantic Analysis", 
            "📊 Statistics", "📖 Document Preview"
        ])
        
        with meta_tab1:
            self.display_overview(metadata)
        
        with meta_tab2:
            self.display_content_analysis(metadata)
        
        with meta_tab3:
            self.display_semantic_analysis(metadata)
        
        with meta_tab4:
            self.display_statistics(metadata)
        
        with meta_tab5:
            self.display_document_preview(document)
        
        # PDF Details tab (if PDF metadata is available)
        if 'pdf_metadata' in metadata:
            with st.expander("PDF Details", expanded=False):
                self.display_pdf_details(metadata['pdf_metadata'])
        
        # Document Relationships tab
        if 'document_relationships' in metadata:
            with st.expander("Document Relationships", expanded=False):
                self.display_document_relationships(metadata['document_relationships'])
        
        # Structural Elements tab
        if 'structural_elements' in metadata:
            with st.expander("Structural Elements", expanded=False):
                self.display_structural_elements(metadata['structural_elements'])

    def display_overview(self, metadata):
        """Display basic file information and summary"""
        st.subheader("📄 File Information")
        
        file_info = metadata['file_info']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", file_info['filename'])
            st.metric("File Size", format_file_size(file_info['file_size']))
        
        with col2:
            st.metric("File Type", file_info['file_type'])
            st.metric("Upload Time", file_info['upload_timestamp'][:19].replace('T', ' '))
        
        with col3:
            st.metric("Processing Time", f"{metadata['processing_info']['processing_time']:.2f}s")
            st.metric("Language", metadata['content_analysis']['language'])
        
        # Document summary
        st.subheader("📝 Document Summary")
        if 'summary' in metadata['semantic_analysis']:
            st.write(metadata['semantic_analysis']['summary'])
        else:
            st.info("No summary available")
        
        # Document classification
        st.subheader("📂 Document Classification")
        if 'document_type' in metadata['semantic_analysis']:
            st.info(f"**Document Type:** {metadata['semantic_analysis']['document_type']}")

    def display_content_analysis(self, metadata):
        """Display content analysis results"""
        content_analysis = metadata['content_analysis']
        
        st.subheader("📊 Content Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", content_analysis['word_count'])
        with col2:
            st.metric("Character Count", content_analysis['character_count'])
        with col3:
            st.metric("Paragraph Count", content_analysis['paragraph_count'])
        with col4:
            st.metric("Sentence Count", content_analysis.get('sentence_count', 'N/A'))
        
        # Readability metrics
        st.subheader("📖 Readability Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'readability_score' in content_analysis:
                score = content_analysis['readability_score']
                st.metric("Readability Score", f"{score:.2f}")
                
                # Readability interpretation
                if score >= 90:
                    level = "Very Easy"
                    color = "green"
                elif score >= 80:
                    level = "Easy"
                    color = "lightgreen"
                elif score >= 70:
                    level = "Fairly Easy"
                    color = "yellow"
                elif score >= 60:
                    level = "Standard"
                    color = "orange"
                else:
                    level = "Difficult"
                    color = "red"
                
                st.markdown(f"**Reading Level:** :{color}[{level}]")
        
        with col2:
            if 'sentiment_score' in content_analysis:
                sentiment = content_analysis['sentiment_score']
                st.metric("Sentiment Score", f"{sentiment:.3f}")
                
                if sentiment > 0.1:
                    st.markdown("**Sentiment:** :green[Positive]")
                elif sentiment < -0.1:
                    st.markdown("**Sentiment:** :red[Negative]")
                else:
                    st.markdown("**Sentiment:** :gray[Neutral]")

    def display_semantic_analysis(self, metadata):
        """Display semantic analysis results"""
        semantic_analysis = metadata['semantic_analysis']
        
        # Keywords
        st.subheader("🔑 Keywords")
        if 'keywords' in semantic_analysis and semantic_analysis['keywords']:
            # Display keywords as tags
            keywords_html = ""
            for keyword in semantic_analysis['keywords'][:15]:  # Limit to top 15
                keywords_html += f'<span style="background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 12px; display: inline-block;">{keyword}</span>'
            st.markdown(keywords_html, unsafe_allow_html=True)
        else:
            st.info("No keywords extracted")
        
        # Key phrases
        st.subheader("💬 Key Phrases")
        if 'key_phrases' in semantic_analysis and semantic_analysis['key_phrases']:
            for i, phrase in enumerate(semantic_analysis['key_phrases'][:10], 1):
                st.write(f"{i}. {phrase}")
        else:
            st.info("No key phrases extracted")
        
        # Named entities
        st.subheader("🏷️ Named Entities")
        if 'named_entities' in semantic_analysis and semantic_analysis['named_entities']:
            entities_df = pd.DataFrame(semantic_analysis['named_entities'])
            if not entities_df.empty:
                st.dataframe(entities_df, use_container_width=True)
            else:
                st.info("No named entities found")
        else:
            st.info("No named entities found")
        
        # Topics
        st.subheader("📚 Topics")
        if 'topics' in semantic_analysis and semantic_analysis['topics']:
            for i, topic in enumerate(semantic_analysis['topics'][:5], 1):
                st.write(f"**Topic {i}:** {topic}")
        else:
            st.info("No topics identified")

    def display_statistics(self, metadata):
        """Display statistical visualizations"""
        st.subheader("📊 Document Statistics")
        
        content_analysis = metadata['content_analysis']
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Word distribution chart
            if 'word_frequency' in content_analysis:
                word_freq = content_analysis['word_frequency']
                if word_freq:
                    df_words = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
                    df_words = df_words.head(15)  # Top 15 words
                    
                    fig = px.bar(df_words, x='Frequency', y='Word', orientation='h',
                               title="Top 15 Most Frequent Words")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Content composition pie chart
            composition_data = {
                'Words': content_analysis['word_count'],
                'Characters': content_analysis['character_count'] - content_analysis['word_count'],
                'Spaces': content_analysis['word_count'] - 1 if content_analysis['word_count'] > 0 else 0
            }
            
            fig = px.pie(values=list(composition_data.values()), 
                        names=list(composition_data.keys()),
                        title="Content Composition")
            st.plotly_chart(fig, use_container_width=True)
        
        # Processing metrics
        st.subheader("⚡ Processing Metrics")
        processing_info = metadata['processing_info']
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Processing Time", f"{processing_info['processing_time']:.2f}s")
        with metrics_col2:
            st.metric("NLP Model", processing_info.get('nlp_model', 'spaCy'))
        with metrics_col3:
            st.metric("Text Length", f"{len(metadata.get('original_text', ''))}")

    def display_document_preview(self, document):
        """Display document preview"""
        st.subheader("📖 Document Preview")
        
        if document and 'text' in document:
            st.text_area(
                "Document Content (Preview)",
                value=document['text'],
                height=400,
                disabled=True
            )
        else:
            st.info("No document preview available")

    def metadata_to_csv(self, metadata):
        """Convert metadata to CSV format"""
        flattened_data = {}
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, list):
                    if value and isinstance(value[0], str):
                        flattened_data[new_key] = '; '.join(map(str, value))
                    else:
                        flattened_data[new_key] = str(value)
                else:
                    flattened_data[new_key] = value
        
        flatten_dict(metadata)
        df = pd.DataFrame([flattened_data])
        return df.to_csv(index=False)

    def analytics_dashboard(self):
        """Display analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.processing_history:
            st.info("📤 No processing history available. Process some documents to see analytics.")
            return
        
        # Processing history analytics
        history_df = pd.DataFrame(st.session_state.processing_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(history_df))
        
        with col2:
            avg_processing_time = history_df['processing_time'].mean()
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
        
        with col3:
            total_size = history_df['file_size'].sum()
            st.metric("Total Data Processed", format_file_size(total_size))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # File type distribution
            file_type_counts = history_df['file_type'].value_counts()
            fig = px.pie(values=file_type_counts.values, names=file_type_counts.index,
                        title="File Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Processing time over time
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            fig = px.line(history_df, x='timestamp', y='processing_time',
                         title="Processing Time Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Processing history table
        st.subheader("📋 Processing History")
        st.dataframe(
            history_df[['filename', 'file_type', 'file_size', 'processing_time', 'timestamp']],
            use_container_width=True
        )
    
    def display_pdf_details(self, pdf_metadata):
        """Display PDF-specific metadata details"""
        st.subheader("📄 PDF Technical Details")
        
        # Technical properties
        tech_props = pdf_metadata.get('technical_properties', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Page Count", tech_props.get('page_count', 0))
        with col2:
            st.metric("Embedded Files", tech_props.get('embedded_files_count', 0))
        with col3:
            st.metric("Form Fields", tech_props.get('form_fields_count', 0))
        
        # Creation info
        creation_info = pdf_metadata.get('creation_info', {})
        if any(creation_info.values()):
            st.subheader("📝 Creation Information")
            if creation_info.get('author'):
                st.write(f"**Author:** {creation_info['author']}")
            if creation_info.get('creator_application'):
                st.write(f"**Created with:** {creation_info['creator_application']}")
            if creation_info.get('creation_date'):
                st.write(f"**Created:** {creation_info['creation_date']}")
        
        # Document structure
        structure = pdf_metadata.get('structure', {})
        if structure.get('bookmarks_count', 0) > 0:
            st.subheader("📚 Document Structure")
            st.write(f"**Bookmarks:** {structure['bookmarks_count']}")
            if structure.get('bookmarks'):
                st.write("**Table of Contents:**")
                for bookmark in structure['bookmarks'][:5]:
                    st.write(f"{'  ' * bookmark.get('level', 1)}- {bookmark.get('title', 'Unknown')}")
        
        # Security information
        if 'document_security' in pdf_metadata:
            security = pdf_metadata['document_security']
            if security.get('is_encrypted') or security.get('has_digital_signatures'):
                st.subheader("🔒 Security Features")
                if security.get('is_encrypted'):
                    st.write("🔐 Document is encrypted")
                if security.get('has_digital_signatures'):
                    st.write("✍️ Contains digital signatures")
    
    def display_document_relationships(self, relationships):
        """Display document relationship analysis"""
        st.subheader("🔗 Document Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if relationships.get('references_other_documents'):
                st.success(f"📚 References other documents ({relationships.get('citation_count', 0)} citations)")
            else:
                st.info("📚 No external references found")
            
            if relationships.get('is_part_of_series'):
                st.success("📖 Part of a document series")
                if relationships.get('series_indicators'):
                    st.write("**Series indicators:**")
                    for indicator in relationships['series_indicators'][:3]:
                        st.write(f"- {indicator}")
            else:
                st.info("📖 Standalone document")
        
        with col2:
            if relationships.get('has_appendices'):
                st.success("📎 Contains appendices")
            else:
                st.info("📎 No appendices found")
            
            if relationships.get('cross_references'):
                st.success(f"🔗 Internal cross-references ({len(relationships['cross_references'])})")
                if relationships['cross_references']:
                    st.write("**Examples:**")
                    for ref in relationships['cross_references'][:3]:
                        st.write(f"- {ref}")
            else:
                st.info("🔗 No internal cross-references")
    
    def display_structural_elements(self, elements):
        """Display structural elements analysis"""
        st.subheader("🏗️ Structural Elements")
        
        # Create metrics for different element types
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tables", elements.get('tables', {}).get('count', 0))
            st.metric("Figures", elements.get('figures', {}).get('count', 0))
        
        with col2:
            st.metric("Equations", elements.get('equations', {}).get('count', 0))
            st.metric("Code Blocks", elements.get('code_blocks', {}).get('count', 0))
        
        with col3:
            st.metric("Lists", elements.get('lists', {}).get('count', 0))
            st.metric("Footnotes", elements.get('footnotes', {}).get('count', 0))
        
        with col4:
            st.metric("Headers", elements.get('headers', {}).get('count', 0))
            if elements.get('headers', {}).get('levels'):
                st.write(f"Levels: {', '.join(map(str, elements['headers']['levels']))}")
        
        # Show examples of found elements
        for element_type, element_data in elements.items():
            if element_data.get('count', 0) > 0 and element_data.get('indicators'):
                with st.expander(f"{element_type.title()} Examples"):
                    for indicator in element_data['indicators'][:5]:
                        st.code(str(indicator))

if __name__ == "__main__":
    app = MetadataApp()
    app.main()
