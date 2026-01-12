import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

class LENADailySummarizer:
    def __init__(self, data_folder="lena_extraction_output"):
        """
        Initialize the LENA Daily Summarizer.
        """
        self.data_folder = data_folder
        self.recording_info = None
        self.segments = None
        self.conversations = None
        self.daily_summary = None
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['font.size'] = 12

    def load_data(self):
        """Load all necessary CSV files."""
        print(f"Loading data from {self.data_folder}...")
        try:
            self.recording_info = pd.read_csv(os.path.join(self.data_folder, "recording_info.csv"))
            self.segments = pd.read_csv(os.path.join(self.data_folder, "segments.csv"))
            self.conversations = pd.read_csv(os.path.join(self.data_folder, "conversations.csv"))
            print("✓ Loaded recording info, segments, and conversations.")
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure all CSV files are in the '{self.data_folder}' directory.")
            return False

    def calculate_daily_totals(self):
        """Calculate the total AWC, CVC, and CTC for each day of the recording."""
        if self.recording_info is None:
            return False

        start_time_column = 'start_clock_time'
        try:
            recording_start_time = pd.to_datetime(self.recording_info[start_time_column].iloc[0])
            print(f"Recording start time found: {recording_start_time}")
        except (KeyError, IndexError):
            print(f"Error: Column '{start_time_column}' not found in recording_info.csv.")
            return False

        daily_awc_cvc = pd.DataFrame()
        daily_ctc = pd.DataFrame()

        # Process Segments
        if self.segments is not None:
            segments_df = self.segments.copy()
            # FIX: Use pd.to_timedelta to correctly parse duration strings like 'PT1.51S'
            segments_df['time_offset'] = pd.to_timedelta(segments_df['start_time'], errors='coerce')
            segments_df.dropna(subset=['time_offset'], inplace=True)
            
            # The calculation is now a simple addition
            segments_df['absolute_time'] = segments_df['time_offset'].apply(lambda x: recording_start_time + x)
            
            segments_df['date'] = segments_df['absolute_time'].dt.date
            segments_df['AWC'] = segments_df['fem_adult_word_cnt'].fillna(0) + segments_df['male_adult_word_cnt'].fillna(0)
            daily_awc_cvc = segments_df.groupby('date').agg(AWC=('AWC', 'sum'), CVC=('child_utt_count', 'sum')).reset_index()

        # Process Conversations
        if self.conversations is not None:
            conv_df = self.conversations.copy()
            # FIX: Use pd.to_timedelta for conversations as well
            conv_df['time_offset'] = pd.to_timedelta(conv_df['start_time'], errors='coerce')
            conv_df.dropna(subset=['time_offset'], inplace=True)
            
            # Simple addition for conversation times
            conv_df['absolute_time'] = conv_df['time_offset'].apply(lambda x: recording_start_time + x)

            conv_df['date'] = conv_df['absolute_time'].dt.date
            daily_ctc = conv_df.groupby('date').agg(CTC=('turn_count', 'sum')).reset_index()

        # Merge final results
        if not daily_awc_cvc.empty:
            self.daily_summary = daily_awc_cvc
            if not daily_ctc.empty:
                self.daily_summary = pd.merge(self.daily_summary, daily_ctc, on='date', how='outer').fillna(0)
            else:
                self.daily_summary['CTC'] = 0
        elif not daily_ctc.empty:
            self.daily_summary = daily_ctc
            self.daily_summary['AWC'] = 0
            self.daily_summary['CVC'] = 0
        else:
            print("No data available to create a daily summary.")
            return False

        print("\nSuccessfully calculated daily totals:")
        print(self.daily_summary)
        return True

    def plot_daily_summary(self, output_dir="lena_visualizations"):
        """Creates and saves a bar chart of the daily AWC, CVC, and CTC."""
        if self.daily_summary is None or self.daily_summary.empty:
            print("No daily summary data to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots()
        self.daily_summary.set_index('date').plot(
            kind='bar', y=['AWC', 'CVC', 'CTC'], ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        ax.set_title('Total AWC, CVC, and CTC per Day', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Total Count', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(['Adult Word Count (AWC)', 'Child Vocalization Count (CVC)', 'Conversational Turn Count (CTC)'])
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', label_type='edge', fontweight='bold')
        plt.tight_layout()
        output_path = os.path.join(output_dir, "daily_summary_AWC_CVC_CTC.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✓ Successfully saved daily summary plot to:\n{output_path}")

def main():
    """Main function to create the daily summary visualization."""
    summarizer = LENADailySummarizer("lena_extraction_output")
    if not summarizer.load_data():
        return
    if not summarizer.calculate_daily_totals():
        return
    summarizer.plot_daily_summary()
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()