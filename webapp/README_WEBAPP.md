# Structural Health Monitoring Dashboard

A React-based web application for inspecting and analyzing structural health monitoring data from fiber optic interrogator systems.

## Features

### 1. Interrogator Data Viewer
- View high-frequency time-series wavelength data from interrogator files
- Interactive zooming with mouse wheel or pinch gestures
- Synchronized zoom across all channels
- Hover tooltips showing exact time and wavelength values
- Support for multiple channels (WL1, WL2, WL3)

### 2. Output Data Viewer  
- Browse processed data by date/time
- View force, displacement, and wavelength shift measurements
- Interactive plots with zoom and pan capabilities
- Synchronized visualization across all metrics

### 3. Strain Analysis Viewer
- Analyze strain vs wavelength relationships
- Select different span lengths (11cm, 15cm, 19cm, 23cm, 27cm)
- Scatter plots showing correlation between mechanical strain and wavelength shifts
- Individual analysis for each fiber optic channel

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Quick Start (Production Mode)
1. From the project root directory, run:
   ```bash
   ./start_webapp.sh
   ```

   This will automatically:
   - Start the backend API server (port 3001)
   - Start the frontend development server (port 5173) 
   - Open access to your data folders

2. Open your browser to `http://localhost:5173`

### Manual Installation
1. Navigate to the webapp directory:
   ```bash
   cd webapp
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start both servers:
   ```bash
   npm run dev:full
   ```
   
   Or start them separately:
   ```bash
   # Terminal 1 - Backend API
   npm run server
   
   # Terminal 2 - Frontend
   npm run dev
   ```

4. Open your browser and go to `http://localhost:5173`

## Current Implementation Status

### ✅ Production Ready!
The webapp now works in full production mode:
- **Direct data loading** from your file system
- **No file upload dialogs** - data loads automatically when you select files
- **Backend API server** serves data from your existing folder structure
- **Real-time data access** to interrogator-data, output, and strain analysis folders

### Architecture

**Backend Server (Node.js/Express):**
- Runs on `http://localhost:3001`
- Serves data directly from your existing folders:
  - `interrogator-data/` → `/api/interrogator-files`
  - `output/` → `/api/output-dates`, `/api/output-files`, `/api/output-data`
  - Latest output data → `/api/strain-analysis`

**Frontend (React):**
- Runs on `http://localhost:5173`
- Fetches data from backend API
- Shows connection status and handles offline scenarios

## Key Technologies

- **React 18** - UI framework
- **Chart.js + react-chartjs-2** - Interactive charting
- **chartjs-plugin-zoom** - Zoom and pan functionality  
- **Papa Parse** - CSV/TSV parsing
- **React Router** - Navigation
- **Tailwind CSS** - Styling

## Interactive Features

### Zoom Controls
- **Mouse Wheel**: Zoom in/out on charts
- **Pinch Gesture**: Touch zoom on mobile devices
- **Ctrl + Drag**: Pan across the chart
- **Reset Zoom Button**: Return to original view

### Data Visualization
- **Synchronized Zooming**: All charts zoom together
- **Hover Tooltips**: Show precise x,y values
- **High-Frequency Support**: Optimized for large datasets
- **Multiple Channels**: Display up to 3 wavelength channels

### File Organization
- **Intuitive Navigation**: Easy switching between data types
- **Date-based Organization**: Quick access to data by timestamp
- **Configuration Selection**: Choose specific test configurations

## Data Format Support

### Interrogator Files (.txt)
```
Timestamp	Time [s]	WL 1[nm]	WL 2[nm]	WL 3[nm]
2025-08-11T14:03:36Z	6703.00000	1555.06566	1538.02023	1524.99374
```

### Output Files (.csv)
```
Air Pressure (bar),Layers (#),Distance (cm),Force (N),Displacement (mm),...
1.0,12.0,11.0,147,36.37,...
```

## Customization

### Adding New Chart Types
1. Register new Chart.js chart type in component imports
2. Create new chart rendering function
3. Add to the main component render method

### Styling Modifications
- Edit `tailwind.config.js` for theme changes
- Modify component classes for specific styling
- Update `src/index.css` for global styles

## Performance Considerations

- Charts are optimized for large datasets (10k+ points)
- Point radius set to 0 by default for performance
- Hover radius enabled for interactivity
- Synchronized zoom events are debounced

## Browser Support

- Chrome 88+
- Firefox 87+
- Safari 14+
- Edge 88+

## Future Enhancements

1. **Real-time Data**: WebSocket integration for live data streaming
2. **Data Export**: Export filtered/zoomed data to CSV
3. **Statistical Analysis**: Built-in statistical calculations
4. **Multiple File Comparison**: Side-by-side analysis
5. **Customizable Dashboards**: User-configurable layouts
6. **Advanced Filtering**: Filter by pressure, force ranges, etc.

## Troubleshooting

### Common Issues

1. **Charts not loading**: Check that all Chart.js dependencies are installed
2. **File upload not working**: Ensure Papa Parse is correctly imported
3. **Zoom not syncing**: Verify chart refs are properly assigned
4. **Styling issues**: Confirm Tailwind CSS is configured correctly

### Performance Issues

1. **Large files**: Consider implementing data pagination
2. **Slow rendering**: Reduce point density or implement virtual scrolling
3. **Memory usage**: Implement data cleanup when switching files
