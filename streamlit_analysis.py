import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Audio Label Performance Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better button styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 80px;
        white-space: pre-line;
        text-align: center;
        font-size: 12px;
        margin: 2px;
        border-radius: 10px;
    }
    
    .stButton > button:hover {
        border-color: #ff6b6b;
        color: #ff6b6b;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #4CAF50;
        border-color: #4CAF50;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        data = pd.read_csv('final_metadata.csv')
        # Remove error values
        data = data[data['likesCount'] != -1]
        
        # Handle NaN values
        for col in ['commentsCount', 'likesCount', 'videoPlayCount', 'videoViewCount']:
            if col in data.columns:
                data[col] = data[col].fillna(0).astype(float)
        
        # Load follower count data
        try:
            followers_data = pd.read_csv('followers-count.csv')
            # Merge with main data based on username
            data = data.merge(followers_data[['userName', 'followersCount']], 
                            left_on='ownerUsername', right_on='userName', how='left')
            # Clean up the merge
            data = data.drop('userName', axis=1)
            # Convert followersCount to numeric, handle any non-numeric values
            data['followersCount'] = pd.to_numeric(data['followersCount'], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            st.warning(f"Could not load follower count data: {e}")
            data['followersCount'] = 0
        
        # Calculate derived metrics
        data['Total_Engagement'] = data['likesCount'] + data['commentsCount']
        data['Replay_Ratio'] = np.where(data['videoViewCount'] > 0, 
                                       data['videoPlayCount'] / data['videoViewCount'], 0)
        
        # Calculate creator-average-based engagement rate (likes / creator's avg views)
        creator_avg_views = data.groupby('ownerUsername')['videoViewCount'].transform('mean')
        data['Engagement_Rate'] = np.where(creator_avg_views > 0,
                                         data['likesCount'] / creator_avg_views * 100, 0)
        
        # Calculate follower-normalized metrics (engagement per 1000 followers)
        data['Likes_Per_1K_Followers'] = np.where(data['followersCount'] > 0,
                                                  data['likesCount'] / (data['followersCount'] / 1000), 0)
        data['Comments_Per_1K_Followers'] = np.where(data['followersCount'] > 0,
                                                     data['commentsCount'] / (data['followersCount'] / 1000), 0)
        data['Views_Per_1K_Followers'] = np.where(data['followersCount'] > 0,
                                                  data['videoViewCount'] / (data['followersCount'] / 1000), 0)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_filtered_data(data, exclude_skewed=True):
    """Get data with option to exclude skewed creators"""
    if exclude_skewed:
        # Exclude problematic creators that skew the analysis
        skewed_creators = ['thalasya_', 'noonoouri']
        return data[~data['ownerUsername'].isin(skewed_creators)]
    return data

def get_creator_stats(data, creator):
    """Get comprehensive stats for a creator"""
    creator_data = data[data['ownerUsername'] == creator]
    
    # Get follower count (should be the same for all rows of this creator)
    follower_count = creator_data['followersCount'].iloc[0] if len(creator_data) > 0 and 'followersCount' in creator_data.columns else 0
    
    stats = {
        'total_reels': len(creator_data),
        'label_distribution': creator_data['Label'].value_counts().to_dict(),
        'follower_count': follower_count,
        'total_views': creator_data['videoViewCount'].sum(),
        'total_likes': creator_data['likesCount'].sum(),
        'total_comments': creator_data['commentsCount'].sum(),
        'avg_engagement_rate': creator_data['Engagement_Rate'].mean()
    }
    
    return stats

def analyze_top_performers(data, metric, top_n=20):
    """Analyze top performing reels by a specific metric"""
    # Use filtered data to exclude skewed creators
    filtered_data = get_filtered_data(data, exclude_skewed=True)
    top_reels = filtered_data.nlargest(top_n, metric)
    
    # Count by label
    label_counts = top_reels['Label'].value_counts()
    label_percentages = (label_counts / len(top_reels) * 100).round(1)
    
    analysis = {
        'top_reels': top_reels,
        'label_counts': label_counts,
        'label_percentages': label_percentages,
        'dominant_label': label_counts.index[0] if len(label_counts) > 0 else None,
        'dominance_percentage': label_percentages.iloc[0] if len(label_percentages) > 0 else 0
    }
    
    return analysis

def cross_metric_analysis(data, top_n=20):
    """Analyze if top reels in one metric also perform well in others"""
    # Use filtered data to exclude skewed creators
    filtered_data = get_filtered_data(data, exclude_skewed=True)
    
    metrics = {
        'Most Reach': 'videoViewCount',
        'Most Traction': 'likesCount', 
        'Best Engagement Rate': 'Engagement_Rate',
        'Most Addictive': 'Replay_Ratio'
    }
    
    results = {}
    overlap_analysis = {}
    
    # Get top performers for each metric
    for metric_name, column in metrics.items():
        # For Replay Ratio, filter for reels with at least average engagement to avoid misleading results
        if column == 'Replay_Ratio':
            avg_engagement = filtered_data['likesCount'].quantile(0.5)  # median likes as baseline
            metric_filtered_data = filtered_data[filtered_data['likesCount'] >= avg_engagement]
            top_reels = metric_filtered_data.nlargest(top_n, column)
        else:
            top_reels = filtered_data.nlargest(top_n, column)
            
        results[metric_name] = {
            'reels': top_reels,
            'label_distribution': top_reels['Label'].value_counts()
        }
    
    # Analyze overlaps
    comparisons = [
        ('Most Reach', 'Most Traction'),
        ('Most Reach', 'Best Engagement Rate'), 
        ('Most Traction', 'Best Engagement Rate'),
        ('Most Traction', 'Most Addictive'),
        ('Best Engagement Rate', 'Most Addictive')
    ]
    
    for metric1, metric2 in comparisons:
        ids1 = set(results[metric1]['reels'].index)
        ids2 = set(results[metric2]['reels'].index)
        overlap = len(ids1.intersection(ids2))
        overlap_percentage = (overlap / top_n) * 100
        
        overlap_analysis[f"{metric1} vs {metric2}"] = {
            'overlap_count': overlap,
            'overlap_percentage': overlap_percentage
        }
    
    return results, overlap_analysis

def main():
    st.title("🎵 Audio Label Performance Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Main analysis type selection at the top
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Creator-Specific Analysis", "Overall Top Performers", "Cross-Metric Analysis", "Label Comparison"],
        help="Select the type of analysis you want to perform"
    )
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    
    # Add minimum reel threshold control
    min_reels = st.sidebar.slider("Minimum reels per creator", 1, 10, 3, help="Creators with fewer reels may have less reliable statistics")
    
    # Get creators with sufficient data
    creator_counts = data['ownerUsername'].value_counts()
    sufficient_creators = creator_counts[creator_counts >= min_reels].index.tolist()
    
    # Debug information
    st.sidebar.write(f"**Debug Info:**")
    st.sidebar.write(f"Total creators: {len(creator_counts)}")
    st.sidebar.write(f"Creators with ≥{min_reels} reels: {len(sufficient_creators)}")
    
    # Show breakdown by reel count
    reel_breakdown = creator_counts.value_counts().sort_index()
    st.sidebar.write("**Creators by reel count:**")
    for reel_count, num_creators in reel_breakdown.items():
        st.sidebar.write(f"• {reel_count} reels: {num_creators} creators")
    
    # Separate skewed creators for special handling - make usernames consistent
    skewed_creators = ['thalasya_', 'noonoouri']  # Updated to exclude thalasya_
    normal_creators = [c for c in sufficient_creators if c not in skewed_creators]
    available_skewed = [c for c in skewed_creators if c in sufficient_creators]
    
    st.sidebar.write(f"Normal creators: {len(normal_creators)}")
    st.sidebar.write(f"Available skewed: {len(available_skewed)}")
    st.sidebar.write(f"Total to show: {len(normal_creators) + len(available_skewed)}")
    
    if analysis_type == "Creator-Specific Analysis":
        st.header("👤 Creator-Specific Analysis")
        
        # Creator selection with buttons instead of dropdown
        st.subheader("Select Creator to Analyze:")
        st.markdown("*Click on any creator below to view their detailed analysis*")
        
        # Create buttons for normal creators first
        cols_per_row = 4
        all_creators_to_show = normal_creators + available_skewed
        num_rows = (len(all_creators_to_show) + cols_per_row - 1) // cols_per_row
        
        selected_creator = None
        
        # Initialize session state for selected creator if not exists
        if 'selected_creator' not in st.session_state:
            st.session_state.selected_creator = all_creators_to_show[0] if all_creators_to_show else None
        
        creator_idx = 0
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            start_idx = row * cols_per_row
            end_idx = min(start_idx + cols_per_row, len(all_creators_to_show))
            
            for col_idx, creator_pos in enumerate(range(start_idx, end_idx)):
                if creator_pos < len(all_creators_to_show):
                    creator = all_creators_to_show[creator_pos]
                    with cols[col_idx]:
                        # Get creator stats for button display
                        creator_data = data[data['ownerUsername'] == creator]
                        follower_count = creator_data['followersCount'].iloc[0] if len(creator_data) > 0 else 0
                        reel_count = len(creator_data)
                        
                        # Create a more informative button
                        if follower_count > 0:
                            if follower_count >= 1000000:
                                follower_display = f"{follower_count/1000000:.1f}M followers"
                            elif follower_count >= 1000:
                                follower_display = f"{follower_count/1000:.0f}K followers"
                            else:
                                follower_display = f"{follower_count} followers"
                        else:
                            follower_display = "N/A followers"
                        
                        # Add skewed data warning for problematic creators
                        if creator in skewed_creators:
                            button_text = f"⚠️ {creator}\n{reel_count} reels | {follower_display}"
                        else:
                            button_text = f"{creator}\n{reel_count} reels | {follower_display}"
                        
                        # Check if this creator is currently selected
                        is_selected = st.session_state.selected_creator == creator
                        
                        # Use different styling for selected creator
                        if is_selected:
                            if st.button(f"✅ {button_text}", key=f"creator_{creator}", type="primary"):
                                st.session_state.selected_creator = creator
                        else:
                            if st.button(button_text, key=f"creator_{creator}"):
                                st.session_state.selected_creator = creator
        
        # Show warning about skewed creators if any are available
        if available_skewed:
            st.warning("⚠️ **Note**: Creators marked with ⚠️ have skewed data that may not represent typical performance patterns. They are excluded from overall analyses but available here for individual review.")
        
        selected_creator = st.session_state.selected_creator
        
        # Add visual separator
        st.markdown("---")
        
        if selected_creator:
            creator_data = data[data['ownerUsername'] == selected_creator]
            stats = get_creator_stats(data, selected_creator)
            
            # Creator Overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reels", stats['total_reels'])
            with col2:
                if stats['follower_count'] > 0:
                    st.metric("Followers", f"{stats['follower_count']:,}")
                else:
                    st.metric("Followers", "N/A")
            with col3:
                st.metric("Total Views", f"{stats['total_views']:,.0f}")
            with col4:
                st.metric("Avg Engagement Rate", f"{stats['avg_engagement_rate']:.2f}%")
            
            # Additional follower-normalized metrics
            if stats['follower_count'] > 0:
                col1, col2, col3 = st.columns(3)
                avg_likes_per_1k = creator_data['Likes_Per_1K_Followers'].mean()
                avg_comments_per_1k = creator_data['Comments_Per_1K_Followers'].mean()
                avg_views_per_1k = creator_data['Views_Per_1K_Followers'].mean()
                
                with col1:
                    st.metric("Avg Likes per 1K Followers", f"{avg_likes_per_1k:.1f}")
                with col2:
                    st.metric("Avg Comments per 1K Followers", f"{avg_comments_per_1k:.1f}")
                with col3:
                    st.metric("Avg Views per 1K Followers", f"{avg_views_per_1k:.0f}")
            
            # Label Distribution
            st.subheader("📊 Label Distribution")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Reel Count by Label:**")
                for label, count in stats['label_distribution'].items():
                    percentage = (count / stats['total_reels']) * 100
                    st.write(f"• {label}: {count} ({percentage:.1f}%)")
            
            with col2:
                fig_pie = px.pie(
                    values=list(stats['label_distribution'].values()),
                    names=list(stats['label_distribution'].keys()),
                    title="Label Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Top Performers by Metric
            st.subheader("🏆 Top Performing Reels")
            
            metric_choice = st.selectbox(
                "Select Metric",
                ["likesCount", "videoViewCount", "Engagement_Rate", "Replay_Ratio"],
                format_func=lambda x: {
                    "likesCount": "Traction (Likes)",
                    "videoViewCount": "Reach (Views)",
                    "Engagement_Rate": "Engagement Rate",
                    "Replay_Ratio": "Replay Ratio (Addictive Factor)"
                }[x]
            )
            
            top_n = st.slider("Number of top reels to show", 5, 20, 10)
            
            top_reels = creator_data.nlargest(top_n, metric_choice)
            
            # Show top reels table
            base_columns = ['Label', 'videoViewCount', 'likesCount', 'commentsCount', 'Replay_Ratio']
            # Add the selected metric if it's not already in the base columns
            if metric_choice not in base_columns:
                display_columns = ['Label', metric_choice] + [col for col in base_columns if col != 'Label']
            else:
                display_columns = base_columns
            
            st.dataframe(
                top_reels[display_columns].round(2),
                use_container_width=True
            )
            
            # Label performance for this creator
            st.subheader("📈 Label Performance Comparison (Averages per Label)")
            st.write("*The table below shows average performance metrics for each audio label used by this creator*")
            
            label_performance = creator_data.groupby('Label').agg({
                'videoViewCount': 'mean',
                'likesCount': 'mean',
                'commentsCount': 'mean',
                'Engagement_Rate': 'mean',
                'Replay_Ratio': 'mean'
            }).round(2)
            
            # Rename columns for clarity
            label_performance.columns = ['Avg Views (Reach)', 'Avg Likes (Traction)', 'Avg Comments', 'Avg Engagement Rate (%)', 'Avg Replay Ratio']
            
            st.dataframe(label_performance, use_container_width=True)
    
    elif analysis_type == "Overall Top Performers":
        st.header("🌟 Overall Top Performers Analysis")
        st.info("📊 **Note**: This analysis excludes creators with skewed data (thalasya_, noonoouri) for more representative results.")
        
        metric_choice = st.selectbox(
            "Select Performance Metric",
            ["likesCount", "videoViewCount", "Engagement_Rate", "Replay_Ratio"],
            format_func=lambda x: {
                "likesCount": "Most Traction (Likes)",
                "videoViewCount": "Most Reach (Views)",
                "Engagement_Rate": "Best Engagement Rate",
                "Replay_Ratio": "Most Addictive (Replay Ratio)"
            }[x]
        )
        
        top_n = st.slider("Number of top reels to analyze", 10, 50, 20)
        
        analysis = analyze_top_performers(data, metric_choice, top_n)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dominant Label", analysis['dominant_label'])
        with col2:
            st.metric("Dominance %", f"{analysis['dominance_percentage']:.1f}%")
        with col3:
            st.metric("Total Labels", len(analysis['label_counts']))
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                x=analysis['label_counts'].index,
                y=analysis['label_counts'].values,
                title=f"Top {top_n} Reels by {metric_choice} - Label Distribution",
                labels={'x': 'Audio Label', 'y': 'Number of Reels'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=analysis['label_percentages'].values,
                names=analysis['label_percentages'].index,
                title="Percentage Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top reels table
        st.subheader(f"🏆 Top {top_n} Reels Details")
        
        # For Engagement Rate metric, add creator average views for clarity
        if metric_choice == 'Engagement_Rate':
            st.info("💡 **Engagement Rate Formula**: (Likes of this reel ÷ Creator's Average Views) × 100")
            
            # Calculate creator average views for display
            display_df = analysis['top_reels'].copy()
            creator_avg_views = data.groupby('ownerUsername')['videoViewCount'].mean()
            display_df['Creator_Avg_Views'] = display_df['ownerUsername'].map(creator_avg_views).round(0)
            
            # Select columns to display for engagement rate
            display_columns = ['ownerUsername', 'followersCount', 'Label', 'videoViewCount', 'likesCount', 'Engagement_Rate', 'Creator_Avg_Views', 'commentsCount', 'Replay_Ratio']
            display_df = display_df[display_columns].copy()
        else:
            # For other metrics
            base_columns = ['ownerUsername', 'followersCount', 'Label', 'videoViewCount', 'likesCount', 'commentsCount', 'Replay_Ratio']
            # Add the selected metric if it's not already in the base columns
            if metric_choice not in base_columns:
                display_columns = ['ownerUsername', 'followersCount', 'Label', metric_choice] + [col for col in base_columns if col not in ['ownerUsername', 'followersCount', 'Label']]
            else:
                display_columns = base_columns
            display_df = analysis['top_reels'][display_columns].copy()
        
        # Format follower count with commas
        if 'followersCount' in display_df.columns:
            display_df['followersCount'] = display_df['followersCount'].apply(lambda x: f"{x:,}" if x > 0 else "N/A")
        
        # Format Creator_Avg_Views if it exists
        if 'Creator_Avg_Views' in display_df.columns:
            display_df['Creator_Avg_Views'] = display_df['Creator_Avg_Views'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_df.round(2),
            use_container_width=True
        )
    
    elif analysis_type == "Cross-Metric Analysis":
        st.header("🔄 Cross-Metric Performance Analysis")
        st.write("This analysis shows how often top performers in one metric also excel in others.")
        st.info("📊 **Note**: This analysis excludes creators with skewed data (thalasya_, noonoouri) for more representative results.")
        
        top_n = st.slider("Number of top reels per metric", 10, 50, 20)
        
        results, overlap_analysis = cross_metric_analysis(data, top_n)
        
        # Overlap Analysis
        st.subheader("📊 Cross-Metric Overlap Analysis")
        
        overlap_data = []
        for comparison, stats in overlap_analysis.items():
            overlap_data.append({
                'Comparison': comparison,
                'Overlap Count': stats['overlap_count'],
                'Overlap Percentage': f"{stats['overlap_percentage']:.1f}%"
            })
        
        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True)
        
        # Individual metric analysis
        st.subheader("🎯 Label Dominance by Metric")
        
        col1, col2 = st.columns(2)
        
        metrics_summary = []
        for metric_name, result in results.items():
            dominant_label = result['label_distribution'].index[0]
            dominance_pct = (result['label_distribution'].iloc[0] / top_n) * 100
            metrics_summary.append({
                'Metric': metric_name,
                'Dominant Label': dominant_label,
                'Dominance %': f"{dominance_pct:.1f}%",
                'Count': result['label_distribution'].iloc[0]
            })
        
        summary_df = pd.DataFrame(metrics_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        # Detailed breakdown
        selected_metric = st.selectbox(
            "Select metric for detailed breakdown",
            list(results.keys())
        )
        
        if selected_metric:
            st.subheader(f"📈 {selected_metric} - Detailed Analysis")
            
            result = results[selected_metric]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bar = px.bar(
                    x=result['label_distribution'].index,
                    y=result['label_distribution'].values,
                    title=f"{selected_metric} - Label Distribution"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Show top reels for this metric
                st.write("**Top Reels:**")
                
                # For Best Engagement Rate, add creator average views for clarity
                if selected_metric == 'Best Engagement Rate':
                    st.info("💡 **Engagement Rate Formula**: (Likes ÷ Creator's Avg Views) × 100")
                    
                    # Calculate creator average views for display
                    display_df = result['reels'].head(10).copy()
                    creator_avg_views = data.groupby('ownerUsername')['videoViewCount'].mean()
                    display_df['Creator_Avg_Views'] = display_df['ownerUsername'].map(creator_avg_views).round(0)
                    
                    # Select only the columns that exist and are needed
                    display_columns = ['ownerUsername', 'followersCount', 'Label', 'videoViewCount', 'likesCount', 'Engagement_Rate', 'Creator_Avg_Views', 'commentsCount', 'Replay_Ratio']
                    display_df = display_df[display_columns]
                else:
                    display_columns = ['ownerUsername', 'followersCount', 'Label', 'videoViewCount', 'likesCount', 'commentsCount', 'Engagement_Rate', 'Replay_Ratio']
                    display_df = result['reels'][display_columns].head(10).copy()
                
                # Format follower count with commas
                if 'followersCount' in display_df.columns:
                    display_df['followersCount'] = display_df['followersCount'].apply(lambda x: f"{x:,}" if x > 0 else "N/A")
                
                # Format Creator_Avg_Views if it exists
                if 'Creator_Avg_Views' in display_df.columns:
                    display_df['Creator_Avg_Views'] = display_df['Creator_Avg_Views'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(
                    display_df.round(2),
                    use_container_width=True
                )
    
    else:  # Label Comparison
        st.header("🏷️ Label Performance Comparison")
        st.info("📊 **Note**: This analysis excludes creators with skewed data (thalasya_, noonoouri) for more representative results.")
        
        # Use filtered data for all label comparison analyses
        filtered_data = get_filtered_data(data, exclude_skewed=True)
        
        # Overall label statistics
        st.subheader("📊 Overall Label Statistics")
        
        label_stats = filtered_data.groupby('Label').agg({
            'videoViewCount': ['count', 'mean', 'max'],
            'likesCount': ['mean', 'max'],
            'commentsCount': ['mean', 'max'],
            'Engagement_Rate': ['mean', 'max'],
            'Replay_Ratio': ['mean', 'max']
        }).round(2)
        
        label_stats.columns = ['_'.join(col).strip() for col in label_stats.columns]
        st.dataframe(label_stats, use_container_width=True)
        
        # Top performing reels by label
        st.subheader("🏆 Best Performing Reel by Label")
        
        metric_for_best = st.selectbox(
            "Select metric to determine 'best'",
            ["videoViewCount", "likesCount", "Engagement_Rate", "Replay_Ratio"],
            format_func=lambda x: {
                "videoViewCount": "Reach (Views)",
                "likesCount": "Traction (Likes)",
                "Engagement_Rate": "Engagement Rate",
                "Replay_Ratio": "Replay Ratio (Addictive Factor)"
            }[x]
        )
        
        best_by_label = filtered_data.loc[filtered_data.groupby('Label')[metric_for_best].idxmax()]
        
        # For Engagement Rate metric, add creator average views for clarity
        if metric_for_best == 'Engagement_Rate':
            st.info("💡 **Engagement Rate Formula**: (Likes of this reel ÷ Creator's Average Views) × 100")
            
            # Calculate creator average views for display
            display_df = best_by_label.copy()
            creator_avg_views = data.groupby('ownerUsername')['videoViewCount'].mean()
            display_df['Creator_Avg_Views'] = display_df['ownerUsername'].map(creator_avg_views).round(0)
            
            # Select only the columns that exist and are needed
            display_columns = ['Label', 'ownerUsername', 'followersCount', 'videoViewCount', 'likesCount', 'Engagement_Rate', 'Creator_Avg_Views', 'commentsCount', 'Replay_Ratio']
            display_df = display_df[display_columns]
        else:
            # For other metrics
            base_columns = ['Label', 'ownerUsername', 'followersCount', 'videoViewCount', 'likesCount', 'commentsCount', 'Replay_Ratio']
            # Add the selected metric if it's not already in the base columns
            if metric_for_best not in base_columns:
                display_columns = ['Label', 'ownerUsername', 'followersCount', metric_for_best] + [col for col in base_columns if col not in ['Label', 'ownerUsername', 'followersCount']]
            else:
                display_columns = base_columns
            display_df = best_by_label[display_columns].copy()
        
        # Format follower count with commas
        if 'followersCount' in display_df.columns:
            display_df['followersCount'] = display_df['followersCount'].apply(lambda x: f"{x:,}" if x > 0 else "N/A")
        
        # Format Creator_Avg_Views if it exists
        if 'Creator_Avg_Views' in display_df.columns:
            display_df['Creator_Avg_Views'] = display_df['Creator_Avg_Views'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_df.round(2),
            use_container_width=True
        )
        
        # Win rate analysis
        st.subheader("🎯 Label Win Rate Analysis")
        st.write("How often each label appears in top performers across different metrics")
        
        win_rates = {}
        metrics_to_analyze = ["videoViewCount", "likesCount", "Engagement_Rate", "Replay_Ratio"]
        
        for metric in metrics_to_analyze:
            top_20 = filtered_data.nlargest(20, metric)
            label_counts = top_20['Label'].value_counts()
            win_rates[metric] = (label_counts / 20 * 100).round(1)
        
        win_rate_df = pd.DataFrame(win_rates).fillna(0)
        st.dataframe(win_rate_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            win_rate_df.T,
            title="Label Win Rates Across Different Metrics (%)",
            labels={'value': 'Win Rate (%)', 'index': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
