import { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import zoomPlugin from 'chartjs-plugin-zoom';
import { 
  getOutputDateFolders, 
  checkAPIHealth 
} from '../utils/dataParser';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

// New API call for combined data
const getCombinedDataForDate = async (date) => {
  try {
    const response = await fetch(`/api/output-data/combined/${date}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch combined data: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error(`Error loading combined data: ${error.message}`);
  }
};

const OutputData = () => {
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [combinedData, setCombinedData] = useState(null);
  const [fileSegments, setFileSegments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const chartRefs = useRef({});

  useEffect(() => {
    checkBackendStatus();
    loadAvailableDates();
  }, []);

  const checkBackendStatus = async () => {
    const isHealthy = await checkAPIHealth();
    setApiStatus(isHealthy ? 'connected' : 'disconnected');
  };

  const loadAvailableDates = async () => {
    try {
      const dates = await getOutputDateFolders();
      setAvailableDates(dates);
    } catch (error) {
      setError(`Failed to load date folders: ${error.message}`);
    }
  };

  const handleDateSelect = async (dateFolder) => {
    setSelectedDate(dateFolder);
    setCombinedData(null);
    setFileSegments([]);
    setError('');
    setLoading(true);
    
    try {
      const { combinedData, fileSegments } = await getCombinedDataForDate(dateFolder);
      setCombinedData(combinedData);
      setFileSegments(fileSegments);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetZoom = () => {
    Object.values(chartRefs.current).forEach(chartRef => {
      if (chartRef && chartRef.resetZoom) {
        chartRef.resetZoom();
      }
    });
  };
  
  const getTimeBasedData = (metric) => {
    if (!combinedData) return null;

    const dataPoints = combinedData.map((point, index) => {
      const value = point[metric];
      if (value !== null && value !== undefined && !isNaN(value)) {
        return {
          x: index / 60, // Convert original index to minutes
          y: value
        };
      }
      return null; // Return null for invalid points
    }).filter(p => p !== null); // Filter out the nulls

    if (dataPoints.length === 0) return null;

    let color, label;
    switch(metric) {
      case 'Force (N)':
        color = 'rgb(30, 144, 255)'; // Dodger Blue
        label = 'Force (N)';
        break;
      case 'Displacement (mm)':
        color = 'rgb(255, 140, 0)'; // Dark Orange
        label = 'Displacement (mm)';
        break;
      case 'WL_ch1':
        color = 'rgb(50, 205, 50)'; // Lime Green
        label = 'WL Ch1';
        break;
      case 'WL_ch2':
        color = 'rgb(220, 20, 60)'; // Crimson
        label = 'WL Ch2';
        break;
      case 'WL_ch3':
        color = 'rgb(148, 0, 211)'; // Dark Violet
        label = 'WL Ch3';
        break;
      default:
        color = 'rgb(128, 128, 128)';
        label = metric;
    }

    return {
      datasets: [
        {
          label: label,
          data: dataPoints,
          borderColor: color,
          backgroundColor: 'transparent',
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 3,
          borderWidth: 1.5,
        },
      ],
    };
  };

  const getCombinedChartOptions = (metric, isLast = false, yMax = 0) => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: false
      },
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = combinedData[Math.round(context.parsed.x * 60)];
            const fileName = point ? point.source_file : 'N/A';
            const precision = metric.includes('WL_') ? 4 : 2;
            return `${context.dataset.label}: ${context.parsed.y.toFixed(precision)} | File: ${fileName}`;
          },
          title: function(context) {
            if (context[0]) {
              return `Time: ${context[0].parsed.x.toFixed(2)} min`;
            }
            return '';
          }
        }
      },
      zoom: {
        limits: {
          x: {min: 'original', max: 'original'},
          y: {min: 'original', max: 'original'}
        },
        pan: {
          enabled: true,
          mode: 'x',
          modifierKey: 'ctrl',
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
          onZoomComplete: function({chart}) {
            // Sync zoom across all charts
            const xScale = chart.scales.x;
            const newMin = xScale.min;
            const newMax = xScale.max;
            
            Object.values(chartRefs.current).forEach(otherChart => {
              if (otherChart && otherChart !== chart) {
                otherChart.zoomScale('x', {min: newMin, max: newMax}, 'none');
              }
            });
          }
        }
      },
    },
    scales: {
      x: {
        type: 'linear',
        display: isLast, // Only show x-axis on the bottom chart
        title: {
          display: isLast,
          text: 'Time (min)',
          font: {
            size: 12
          }
        },
        ticks: {
          display: isLast
        },
        grid: {
          display: isLast
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: metric.includes('WL_') ? 'Wavelength Shift (nm)' : 
                metric.includes('Force') ? 'Force (N)' :
                metric.includes('Displacement') ? 'Displacement (mm)' : metric,
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(0,0,0,0.1)'
        }
      },
    },
  });

  const renderCombinedView = () => {
    if (!combinedData) return null;

    const metrics = ['Force (N)', 'Displacement (mm)', 'WL_ch1', 'WL_ch2', 'WL_ch3'];
    const availableMetrics = metrics.filter(metric => {
      const data = getTimeBasedData(metric);
      return data && data.datasets[0].data.length > 0;
    });

    // Pre-calculate yMax for the Force chart to pass to options
    const forceData = getTimeBasedData('Force (N)');
    const yMax = forceData ? Math.max(...forceData.datasets[0].data.map(p => p.y)) : 0;

    // Create background zones based on data patterns
    const totalTime = combinedData.length / 60; // Convert to minutes
    const backgroundZones = [];
    const zoneSize = totalTime / 8;
    const colors = [
      'rgba(255, 182, 193, 0.15)', // Light pink
      'rgba(152, 251, 152, 0.15)', // Light green  
      'rgba(173, 216, 230, 0.15)', // Light blue
      'rgba(255, 218, 185, 0.15)', // Light orange
      'rgba(221, 160, 221, 0.15)', // Light purple
    ];

    for (let i = 0; i < 8; i++) {
      const startTime = i * zoneSize;
      const endTime = (i + 1) * zoneSize;
      backgroundZones.push({
        start: startTime,
        end: endTime,
        color: colors[i % colors.length],
      });
    }

    return (
      <div className="space-y-1" style={{ position: 'relative' }}>
        {/* Background zones */}
        <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 1 }}>
          {backgroundZones.map((zone, index) => {
            const leftPercent = (zone.start / totalTime) * 100;
            const widthPercent = ((zone.end - zone.start) / totalTime) * 100;
            
            return (
              <div
                key={index}
                className="absolute h-full"
                style={{
                  left: `${leftPercent}%`,
                  width: `${widthPercent}%`,
                  backgroundColor: zone.color,
                  zIndex: 1
                }}
              />
            );
          })}
        </div>
        
        {/* Charts */}
        <div className="relative" style={{ zIndex: 2 }}>
          {availableMetrics.map((metric, index) => {
            const data = getTimeBasedData(metric);
            const isLast = index === availableMetrics.length - 1;
            
            return (
              <div key={metric} className="bg-transparent rounded-lg shadow-sm border border-gray-200 mb-1">
                <div style={{ height: '200px', padding: '10px', position: 'relative' }}>
                  <Line
                    ref={(ref) => {
                      if (ref) {
                        chartRefs.current[`combined_${metric}`] = ref;
                      }
                    }}
                    data={data}
                    options={getCombinedChartOptions(metric, isLast, yMax)}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const formatDate = (dateString) => {
    // Convert YYYYMMDD_HHMMSS to readable format
    const year = dateString.substring(0, 4);
    const month = dateString.substring(4, 6);
    const day = dateString.substring(6, 8);
    const hour = dateString.substring(9, 11);
    const minute = dateString.substring(11, 13);
    const second = dateString.substring(13, 15);
    
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Output Data Viewer</h2>
      
      {/* API Status */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              apiStatus === 'connected' ? 'bg-green-500' : 
              apiStatus === 'disconnected' ? 'bg-red-500' : 
              'bg-yellow-500'
            }`}></div>
            <span className="text-sm font-medium">
              Backend Status: {
                apiStatus === 'connected' ? 'Connected' : 
                apiStatus === 'disconnected' ? 'Disconnected' : 
                'Checking...'
              }
            </span>
          </div>
          {apiStatus === 'disconnected' && (
            <button
              onClick={checkBackendStatus}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Retry
            </button>
          )}
        </div>
      </div>

      {/* Date Selection */}
      {apiStatus === 'connected' && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Select Date/Time</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
            {availableDates.map((dateFolder) => (
              <button
                key={dateFolder}
                onClick={() => handleDateSelect(dateFolder)}
                disabled={loading}
                className={`p-3 text-sm border-2 rounded-lg transition-colors text-left ${
                  selectedDate === dateFolder
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : loading 
                    ? 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed'
                    : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                <div className="font-medium">{formatDate(dateFolder)}</div>
                <div className="text-xs text-gray-500 mt-1">{dateFolder}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Disconnected State */}
      {apiStatus === 'disconnected' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <h3 className="text-lg font-medium text-red-800 mb-2">Backend Not Available</h3>
          <p className="text-red-700 mb-4">
            The data server is not running. Please make sure the backend server is started.
          </p>
          <div className="text-sm text-red-600">
            <p>To start the backend server, run:</p>
            <code className="bg-red-100 px-2 py-1 rounded mt-2 inline-block">npm run dev:full</code>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading data...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Data Display */}
      {combinedData && (
        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Force, Displacement, and Wavelength Shifts (Compressed Timeline)
          </h3>
          {renderCombinedView()}
        </div>
      )}

      {/* Instructions */}
      {!selectedDate && apiStatus === 'connected' && (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <h3 className="text-lg font-medium text-gray-800 mb-2">Get Started</h3>
          <p className="text-gray-600 mb-4">
            Select a date/time above to view the processed output data with force, displacement, and wavelength shift information.
          </p>
          <div className="text-sm text-gray-500">
            <p>Data is loaded directly from your output folder.</p>
          </div>
        </div>
      )}

    </div>
  );
};

export default OutputData;