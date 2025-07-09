import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

class SportsAnalyticsVisualizer:
    """Comprehensive visualization suite for sports analytics data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the analytics data"""
        try:
            df = pd.read_csv(self.csv_path, comment='#')
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                
            # Filter out invalid data
            df = df[df['frame_id'] >= 0]
            
            logging.info(f"Loaded {len(df)} rows of analytics data")
            return df
            
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def create_object_detection_summary(self) -> go.Figure:
        """Create summary visualization of object detection results"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Objects Detected Over Time', 'Detection Confidence Distribution',
                           'Team Distribution', 'Object Type Distribution'),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Objects detected over time
        objects_per_frame = self.df.groupby('frame_id').size()
        fig.add_trace(
            go.Scatter(x=objects_per_frame.index, y=objects_per_frame.values,
                      mode='lines', name='Objects per Frame'),
            row=1, col=1
        )
        
        # Detection confidence distribution
        valid_conf = self.df[self.df['confidence'] > 0]['confidence']
        fig.add_trace(
            go.Histogram(x=valid_conf, nbinsx=20, name='Confidence Distribution'),
            row=1, col=2
        )
        
        # Team distribution
        team_counts = self.df['team'].value_counts()
        fig.add_trace(
            go.Pie(labels=team_counts.index, values=team_counts.values, name='Teams'),
            row=2, col=1
        )
        
        # Object type distribution
        type_counts = self.df['object_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values, name='Object Types'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Sports Analytics Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_player_movement_heatmap(self) -> go.Figure:
        """Create heatmap of player movements on the pitch"""
        if 'pitch_x' not in self.df.columns or 'pitch_y' not in self.df.columns:
            logging.warning("Pitch coordinates not available")
            return go.Figure()
        
        # Filter valid pitch coordinates
        valid_coords = self.df.dropna(subset=['pitch_x', 'pitch_y'])
        
        if len(valid_coords) == 0:
            logging.warning("No valid pitch coordinates found")
            return go.Figure()
        
        fig = go.Figure()
        
        # Create heatmap for each team
        teams = valid_coords['team'].unique()
        for team in teams:
            team_data = valid_coords[valid_coords['team'] == team]
            
            if len(team_data) > 0:
                fig.add_trace(go.Histogram2d(
                    x=team_data['pitch_x'],
                    y=team_data['pitch_y'],
                    name=f'{team} Movement',
                    opacity=0.6,
                    nbinsx=50,
                    nbinsy=30
                ))
        
        fig.update_layout(
            title="Player Movement Heatmap",
            xaxis_title="Pitch X Coordinate",
            yaxis_title="Pitch Y Coordinate",
            height=600
        )
        
        return fig
    
    def create_action_timeline(self) -> go.Figure:
        """Create timeline visualization of detected actions"""
        action_data = self.df[self.df['action_event'] != ''].copy()
        
        if len(action_data) == 0:
            logging.warning("No action events found")
            return go.Figure()
        
        fig = go.Figure()
        
        # Group actions by type
        action_types = action_data['action_event'].unique()
        colors = px.colors.qualitative.Set3[:len(action_types)]
        
        for i, action_type in enumerate(action_types):
            action_subset = action_data[action_data['action_event'] == action_type]
            
            fig.add_trace(go.Scatter(
                x=action_subset['timestamp'],
                y=[action_type] * len(action_subset),
                mode='markers',
                marker=dict(
                    size=action_subset['action_velocity'] / 10,
                    color=colors[i],
                    line=dict(width=1, color='black')
                ),
                name=action_type,
                hovertemplate='<b>%{y}</b><br>' +
                             'Time: %{x:.2f}s<br>' +
                             'Velocity: %{customdata:.2f}<br>' +
                             '<extra></extra>',
                customdata=action_subset['action_velocity']
            ))
        
        fig.update_layout(
            title="Action Events Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Action Type",
            height=400
        )
        
        return fig
    
    def create_tracking_quality_analysis(self) -> go.Figure:
        """Analyze tracking quality over time"""
        quality_data = self.df[self.df['tracking_quality'] > 0].copy()
        
        if len(quality_data) == 0:
            logging.warning("No tracking quality data found")
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Tracking Quality Over Time', 'Quality Distribution by Object Type')
        )
        
        # Quality over time
        quality_over_time = quality_data.groupby('frame_id')['tracking_quality'].mean()
        fig.add_trace(
            go.Scatter(x=quality_over_time.index, y=quality_over_time.values,
                      mode='lines', name='Average Quality'),
            row=1, col=1
        )
        
        # Quality distribution by object type
        for obj_type in quality_data['object_type'].unique():
            type_data = quality_data[quality_data['object_type'] == obj_type]
            fig.add_trace(
                go.Box(y=type_data['tracking_quality'], name=obj_type),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Tracking Quality Analysis",
            height=600
        )
        
        return fig
    
    def export_dashboard(self, output_path: str = "sports_analytics_dashboard.html"):
        """Export complete dashboard to HTML file"""
        try:
            # Create all visualizations
            summary_fig = self.create_object_detection_summary()
            heatmap_fig = self.create_player_movement_heatmap()
            timeline_fig = self.create_action_timeline()
            quality_fig = self.create_tracking_quality_analysis()
            
            # Combine into dashboard
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sports Analytics Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Sports Analytics Dashboard</h1>
                
                <div id="summary-chart" style="width:100%;height:800px;"></div>
                <div id="heatmap-chart" style="width:100%;height:600px;"></div>
                <div id="timeline-chart" style="width:100%;height:400px;"></div>
                <div id="quality-chart" style="width:100%;height:600px;"></div>
                
                <script>
                    Plotly.newPlot('summary-chart', {summary_fig.to_json()});
                    Plotly.newPlot('heatmap-chart', {heatmap_fig.to_json()});
                    Plotly.newPlot('timeline-chart', {timeline_fig.to_json()});
                    Plotly.newPlot('quality-chart', {quality_fig.to_json()});
                </script>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(dashboard_html)
                
            logging.info(f"Dashboard exported to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to export dashboard: {e}")

def main():
    """Main function to generate visualizations"""
    visualizer = SportsAnalyticsVisualizer('enterprise_analytics_output.csv')
    
    # Generate individual plots
    summary_fig = visualizer.create_object_detection_summary()
    summary_fig.show()
    
    heatmap_fig = visualizer.create_player_movement_heatmap()
    heatmap_fig.show()
    
    timeline_fig = visualizer.create_action_timeline()
    timeline_fig.show()
    
    quality_fig = visualizer.create_tracking_quality_analysis()
    quality_fig.show()
    
    # Export dashboard
    visualizer.export_dashboard()

if __name__ == "__main__":
    main()
