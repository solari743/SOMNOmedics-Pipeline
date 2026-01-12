import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

class LENAExtractor:
    def __init__(self, its_file_path):
        """
        Initialize LENA ITS file extractor
        
        Args:
            its_file_path (str): Path to the .its file
        """
        self.its_file_path = its_file_path
        self.tree = None
        self.root = None
        self.recording_info = {}
        self.segments = []
        self.conversations = []
        self.summary_stats = {}
        
    def parse_file(self):
        """Parse the ITS XML file"""
        try:
            print(f"Parsing ITS file: {self.its_file_path}")
            self.tree = ET.parse(self.its_file_path)
            self.root = self.tree.getroot()
            print("File parsed successfully!")
            return True
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return False
        except FileNotFoundError:
            print(f"File not found: {self.its_file_path}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def extract_recording_info(self):
        """Extract recording metadata"""
        print("Extracting recording information...")
        
        # Find ProcessingUnit (contains recording metadata)
        processing_unit = self.root.find('.//ProcessingUnit')
        if processing_unit is not None:
            self.recording_info = {
                'file_path': self.its_file_path,
                'software_version': processing_unit.get('version', 'Unknown'),
                'analysis_date': processing_unit.get('analysisDate', 'Unknown'),
                'analysis_version': processing_unit.get('analysisVersion', 'Unknown')
            }
        
        # Find Recording element
        recording = self.root.find('.//Recording')
        if recording is not None:
            self.recording_info.update({
                'recording_num': recording.get('num', 'Unknown'),
                'start_time': recording.get('startTime', 'Unknown'),
                'end_time': recording.get('endTime', 'Unknown'),
                'start_clock_time': recording.get('startClockTime', 'Unknown'),
                'end_clock_time': recording.get('endClockTime', 'Unknown')
            })
            
            # Calculate duration if possible
            if 'startTime' in self.recording_info and 'endTime' in self.recording_info:
                try:
                    start = float(self.recording_info['startTime'])
                    end = float(self.recording_info['endTime'])
                    duration_seconds = (end - start) / 1000  # Convert from milliseconds
                    self.recording_info['duration_seconds'] = duration_seconds
                    self.recording_info['duration_hours'] = duration_seconds / 3600
                except:
                    pass
        
        # Find child information
        child_info = self.root.find('.//ChildInfo')
        if child_info is not None:
            self.recording_info.update({
                'child_gender': child_info.get('gender', 'Unknown'),
                'child_age_months': child_info.get('ageInMonths', 'Unknown'),
                'child_age_weeks': child_info.get('ageInWeeks', 'Unknown'),
                'child_dob': child_info.get('dob', 'Unknown')
            })
        
        return self.recording_info
    
    def extract_segments(self):
        """Extract all audio segments with detailed information"""
        print("Extracting segment data...")
        
        segments = []
        segment_elements = self.root.findall('.//Segment')
        
        for i, segment in enumerate(segment_elements):
            seg_data = {
                'segment_id': i,
                'start_time': segment.get('startTime', '0'),
                'end_time': segment.get('endTime', '0'),
                'speaker_type': segment.get('spkr', 'Unknown'),
                'start_utt_time': segment.get('startUttTime', '0'),
                'end_utt_time': segment.get('endUttTime', '0'),
                'start_utt_num': segment.get('startUttNum', '0'),
                'end_utt_num': segment.get('endUttNum', '0'),
                'child_utt_count': segment.get('childUttCnt', '0'),
                'child_utt_len': segment.get('childUttLen', '0'),
                'child_cry_vfx_len': segment.get('childCryVfxLen', '0'),
                'child_canont_len': segment.get('childCanontLen', '0'),
                'child_non_speech_len': segment.get('childNonSpeechLen', '0'),
                'fem_adult_word_cnt': segment.get('femAdultWordCnt', '0'),
                'male_adult_word_cnt': segment.get('maleAdultWordCnt', '0'),
                'fem_adult_utt_cnt': segment.get('femAdultUttCnt', '0'),
                'male_adult_utt_cnt': segment.get('maleAdultUttCnt', '0'),
                'fem_adult_utt_len': segment.get('femAdultUttLen', '0'),
                'male_adult_utt_len': segment.get('maleAdultUttLen', '0'),
                'fem_adult_non_speech_len': segment.get('femAdultNonSpeechLen', '0'),
                'male_adult_non_speech_len': segment.get('maleAdultNonSpeechLen', '0'),
                'overlap_cnt': segment.get('overlapCnt', '0'),
                'average_db': segment.get('average_dB', '0'),
                'peak_db': segment.get('peak_dB', '0')
            }
            
            # Calculate duration
            try:
                start = float(seg_data['start_time'])
                end = float(seg_data['end_time'])
                seg_data['duration_ms'] = end - start
                seg_data['duration_seconds'] = (end - start) / 1000
            except:
                seg_data['duration_ms'] = 0
                seg_data['duration_seconds'] = 0
            
            segments.append(seg_data)
        
        self.segments = segments
        print(f"Extracted {len(segments)} segments")
        return segments
    
    def extract_conversations(self):
        """Extract conversation turn information"""
        print("Extracting conversation data...")
        
        conversations = []
        conv_elements = self.root.findall('.//Conversation')
        
        for i, conv in enumerate(conv_elements):
            conv_data = {
                'conversation_id': i,
                'start_time': conv.get('startTime', '0'),
                'end_time': conv.get('endTime', '0'),
                'turn_count': conv.get('turnTaking', '0'),
                'start_utt_num': conv.get('startUttNum', '0'),
                'end_utt_num': conv.get('endUttNum', '0')
            }
            
            # Calculate duration
            try:
                start = float(conv_data['start_time'])
                end = float(conv_data['end_time'])
                conv_data['duration_ms'] = end - start
                conv_data['duration_seconds'] = (end - start) / 1000
            except:
                conv_data['duration_ms'] = 0
                conv_data['duration_seconds'] = 0
            
            conversations.append(conv_data)
        
        self.conversations = conversations
        print(f"Extracted {len(conversations)} conversations")
        return conversations
    
    def calculate_summary_stats(self):
        """Calculate comprehensive summary statistics"""
        print("Calculating summary statistics...")
        
        if not self.segments:
            print("No segments found for summary calculation")
            return {}
        
        df_segments = pd.DataFrame(self.segments)
        
        # Convert string columns to numeric
        numeric_columns = [
            'child_utt_count', 'child_utt_len', 'child_cry_vfx_len', 'child_canont_len',
            'child_non_speech_len', 'fem_adult_word_cnt', 'male_adult_word_cnt',
            'fem_adult_utt_cnt', 'male_adult_utt_cnt', 'fem_adult_utt_len',
            'male_adult_utt_len', 'fem_adult_non_speech_len', 'male_adult_non_speech_len',
            'overlap_cnt', 'duration_seconds'
        ]
        
        for col in numeric_columns:
            if col in df_segments.columns:
                df_segments[col] = pd.to_numeric(df_segments[col], errors='coerce').fillna(0)
        
        # Calculate summary statistics
        summary = {
            'total_recording_time_seconds': df_segments['duration_seconds'].sum(),
            'total_recording_time_hours': df_segments['duration_seconds'].sum() / 3600,
            'total_segments': len(df_segments),
            
            # Adult Word Counts (AWC)
            'adult_word_count_female': df_segments['fem_adult_word_cnt'].sum(),
            'adult_word_count_male': df_segments['male_adult_word_cnt'].sum(),
            'adult_word_count_total': df_segments['fem_adult_word_cnt'].sum() + df_segments['male_adult_word_cnt'].sum(),
            
            # Child Vocalizations (CVC)
            'child_vocalization_count': df_segments['child_utt_count'].sum(),
            'child_vocalization_length_total': df_segments['child_utt_len'].sum(),
            'child_canonical_length': df_segments['child_canont_len'].sum(),
            'child_cry_length': df_segments['child_cry_vfx_len'].sum(),
            'child_non_speech_length': df_segments['child_non_speech_len'].sum(),
            
            # Adult Utterances
            'adult_utterance_count_female': df_segments['fem_adult_utt_cnt'].sum(),
            'adult_utterance_count_male': df_segments['male_adult_utt_cnt'].sum(),
            'adult_utterance_count_total': df_segments['fem_adult_utt_cnt'].sum() + df_segments['male_adult_utt_cnt'].sum(),
            'adult_utterance_length_female': df_segments['fem_adult_utt_len'].sum(),
            'adult_utterance_length_male': df_segments['male_adult_utt_len'].sum(),
            
            # Conversational Turns (CTC)
            'total_conversations': len(self.conversations),
            'total_conversation_time': sum([conv['duration_seconds'] for conv in self.conversations]),
            'total_turn_count': sum([int(conv['turn_count']) for conv in self.conversations if conv['turn_count'].isdigit()]),
            
            # Overlap information
            'total_overlaps': df_segments['overlap_cnt'].sum(),
            
            # Speaker distribution
            'speaker_type_distribution': df_segments['speaker_type'].value_counts().to_dict()
        }
        
        # Calculate rates per hour
        hours = summary['total_recording_time_hours']
        if hours > 0:
            summary.update({
                'adult_words_per_hour': summary['adult_word_count_total'] / hours,
                'child_vocalizations_per_hour': summary['child_vocalization_count'] / hours,
                'conversations_per_hour': summary['total_conversations'] / hours,
                'turns_per_hour': summary['total_turn_count'] / hours
            })
        
        self.summary_stats = summary
        return summary
    
    def create_hourly_analysis(self):
        """Create hour-by-hour breakdown of activity"""
        print("Creating hourly analysis...")
        
        if not self.segments:
            return pd.DataFrame()
        
        df_segments = pd.DataFrame(self.segments)
        
        # Convert to numeric
        numeric_cols = ['start_time', 'fem_adult_word_cnt', 'male_adult_word_cnt', 'child_utt_count']
        for col in numeric_cols:
            df_segments[col] = pd.to_numeric(df_segments[col], errors='coerce').fillna(0)
        
        # Create hour bins
        df_segments['hour'] = (df_segments['start_time'] / (1000 * 60 * 60)).astype(int)
        
        # Group by hour
        hourly_stats = df_segments.groupby('hour').agg({
            'fem_adult_word_cnt': 'sum',
            'male_adult_word_cnt': 'sum',
            'child_utt_count': 'sum',
            'duration_seconds': 'sum'
        }).reset_index()
        
        hourly_stats['total_adult_words'] = hourly_stats['fem_adult_word_cnt'] + hourly_stats['male_adult_word_cnt']
        
        return hourly_stats
    
    def save_all_data(self, output_dir="lena_extraction_output"):
        """Save all extracted data to CSV files"""
        print(f"Saving data to {output_dir}/...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save recording info
        if self.recording_info:
            pd.DataFrame([self.recording_info]).to_csv(
                os.path.join(output_dir, "recording_info.csv"), index=False
            )
        
        # Save segments
        if self.segments:
            pd.DataFrame(self.segments).to_csv(
                os.path.join(output_dir, "segments.csv"), index=False
            )
        
        # Save conversations
        if self.conversations:
            pd.DataFrame(self.conversations).to_csv(
                os.path.join(output_dir, "conversations.csv"), index=False
            )
        
        # Save summary stats
        if self.summary_stats:
            pd.DataFrame([self.summary_stats]).to_csv(
                os.path.join(output_dir, "summary_statistics.csv"), index=False
            )
        
        # Save hourly analysis
        hourly_df = self.create_hourly_analysis()
        if not hourly_df.empty:
            hourly_df.to_csv(
                os.path.join(output_dir, "hourly_analysis.csv"), index=False
            )
        
        print(f"All data saved to {output_dir}/")
        return output_dir
    
    def print_summary(self):
        """Print a summary of extracted data"""
        print("\n" + "="*50)
        print("LENA ITS FILE EXTRACTION SUMMARY")
        print("="*50)
        
        if self.recording_info:
            duration = self.recording_info.get('duration_hours', 'Unknown')
            if duration != 'Unknown' and duration is not None:
                print(f"Recording Duration: {float(duration):.2f} hours")
            else:
                print(f"Recording Duration: Unknown")
            print(f"Child Age: {self.recording_info.get('child_age_months', 'Unknown')} months")
            print(f"Child Gender: {self.recording_info.get('child_gender', 'Unknown')}")
        
        if self.summary_stats:
            print(f"\nKey Metrics:")
            print(f"  Adult Word Count (AWC): {self.summary_stats.get('adult_word_count_total', 0):,}")
            print(f"  Child Vocalizations (CVC): {self.summary_stats.get('child_vocalization_count', 0):,}")
            print(f"  Conversational Turns (CTC): {self.summary_stats.get('total_turn_count', 0):,}")
            print(f"  Total Conversations: {self.summary_stats.get('total_conversations', 0):,}")
            
            if 'adult_words_per_hour' in self.summary_stats:
                print(f"\nRates per Hour:")
                print(f"  Adult Words/Hour: {self.summary_stats['adult_words_per_hour']:.1f}")
                print(f"  Child Vocalizations/Hour: {self.summary_stats['child_vocalizations_per_hour']:.1f}")
                print(f"  Conversations/Hour: {self.summary_stats['conversations_per_hour']:.1f}")
        
        print(f"\nData Extracted:")
        print(f"  Segments: {len(self.segments):,}")
        print(f"  Conversations: {len(self.conversations):,}")
        print("="*50)

def main():
    """Main function to run the extractor"""
    # PUT YOUR FILE PATH HERE:
    its_file_path = r"c:\Users\SOMNOmedics\Desktop\DATA - POSTER\DATASET - HIGH RISK 02\High_Risk02 [LENA]\20250107_115931_063914.its"
    
    # Initialize extractor
    extractor = LENAExtractor(its_file_path)
    
    # Parse file
    if not extractor.parse_file():
        return
    
    # Extract all data
    extractor.extract_recording_info()
    extractor.extract_segments()
    extractor.extract_conversations()
    extractor.calculate_summary_stats()
    
    # Print summary
    extractor.print_summary()
    
    # Save all data
    output_dir = extractor.save_all_data()
    
    print(f"\nExtraction complete! Check the '{output_dir}' folder for all CSV files.")

if __name__ == "__main__":
    main()