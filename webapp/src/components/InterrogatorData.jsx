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
  TimeScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import zoomPlugin from 'chartjs-plugin-zoom';
import { parseInterrogatorFile, getInterrogatorFileList, checkAPIHealth } from '../utils/dataParser';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  zoomPlugin
);

const InterrogatorData = () => {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [fileData, setFileData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const chartRefs = useRef([]);

  useEffect(() => {
    checkBackendStatus();
    loadAvailableFiles();
  }, []);

  const checkBackendStatus = async () => {
    const isHealthy = await checkAPIHealth();
    setApiStatus(isHealthy ? 'connected' : 'disconnected');
  };

  const loadAvailableFiles = async () => {
    try {
      const files = await getInterrogatorFileList();
      setAvailableFiles(files);
    } catch (error) {
      setError(`Failed to load file list: ${error.message}`);
    }
  };

  const handleFileSelect = async (filename) => {
    setSelectedFile(filename);
    setError('');
    setLoading(true);

    try {
      const parsedData = await parseInterrogatorFile(filename);
      setFileData(parsedData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetZoom = () => {
    chartRefs.current.forEach(chartRef => {
      if (chartRef && chartRef.resetZoom) {
        chartRef.resetZoom();
      }
    });
  };

  const getChartData = (channelNumber) => {
    if (!fileData) return null;

    const channelKey = `wl${channelNumber}`;
    const validData = fileData.data.filter(point => point[channelKey] !== null);

    return {
      datasets: [
        {
          label: `Wavelength Channel ${channelNumber} [nm]`,
          data: validData.map(point => ({
            x: point.time,
            y: point[channelKey]
          })),
          borderColor: channelNumber === 1 ? 'rgb(255, 99, 132)' : 
                      channelNumber === 2 ? 'rgb(54, 162, 235)' : 
                      'rgb(255, 205, 86)',
          backgroundColor: channelNumber === 1 ? 'rgba(255, 99, 132, 0.2)' : 
                          channelNumber === 2 ? 'rgba(54, 162, 235, 0.2)' : 
                          'rgba(255, 205, 86, 0.2)',
          tension: 0,
          pointRadius: 0,
          pointHoverRadius: 3,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      title: {
        display: false,
      },
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(6)} nm`;
          },
          title: function(context) {
            return `Time: ${context[0].parsed.x.toFixed(3)} s`;
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
            
            chartRefs.current.forEach(otherChart => {
              if (otherChart && otherChart !== chart) {
                otherChart.zoomScale('x', {min: newMin, max: newMax}, 'none');
              }
            });
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        display: true,
        title: {
          display: true,
          text: 'Time (s)',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Wavelength (nm)',
        },
      },
    },
  };

  const renderChannelChart = (channelNumber) => {
    const data = getChartData(channelNumber);
    if (!data || !data.datasets[0].data.length) return null;

    return (
      <div key={channelNumber} className="bg-white rounded-lg shadow p-4 mb-4">
        <h3 className="text-lg font-semibold mb-3">Channel {channelNumber}</h3>
        <div style={{ height: '300px' }}>
          <Line
            ref={(ref) => {
              if (ref) {
                chartRefs.current[channelNumber - 1] = ref;
              }
            }}
            data={data}
            options={chartOptions}
          />
        </div>
      </div>
    );
  };

  const formatFileName = (filename) => {
    return filename.replace('-interrogator.txt', '').replace(/-/g, ' ');
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Interrogator Data Viewer</h2>
      
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

      {/* File Selection */}
      {apiStatus === 'connected' && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Select Data File</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 mb-4">
            {availableFiles.map((filename) => (
              <button
                key={filename}
                onClick={() => handleFileSelect(filename)}
                disabled={loading}
                className={`p-3 text-sm border-2 rounded-lg transition-colors ${
                  selectedFile === filename
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : loading 
                    ? 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed'
                    : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                {formatFileName(filename)}
              </button>
            ))}
          </div>
          
          {selectedFile && (
            <div className="text-sm text-gray-600">
              Selected: <span className="font-medium">{formatFileName(selectedFile)}</span>
            </div>
          )}
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
      {fileData && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-600">
                <strong>File:</strong> {formatFileName(fileData.filename)} | 
                <strong> Data Points:</strong> {fileData.data.length.toLocaleString()} | 
                <strong> Channels:</strong> {fileData.channels}
              </div>
              <div className="space-x-2">
                <button
                  onClick={resetZoom}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                >
                  Reset Zoom
                </button>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              <strong>Zoom:</strong> Mouse wheel or pinch | <strong>Pan:</strong> Ctrl + Drag
            </div>
          </div>

          {/* Charts */}
          <div className="space-y-4">
            {[1, 2, 3].map(channelNumber => renderChannelChart(channelNumber))}
          </div>
        </div>
      )}

      {/* Instructions */}
      {!fileData && !loading && apiStatus === 'connected' && (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <h3 className="text-lg font-medium text-gray-800 mb-2">Get Started</h3>
          <p className="text-gray-600 mb-4">
            Select a data file above to view the interrogator wavelength data with interactive zoom and hover capabilities.
          </p>
          <div className="text-sm text-gray-500">
            <p>Data is loaded directly from your interrogator-data folder.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InterrogatorData;