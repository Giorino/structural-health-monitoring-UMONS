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
import { Scatter } from 'react-chartjs-2';
import zoomPlugin from 'chartjs-plugin-zoom';
import { 
  parseStrainAnalysisFile, 
  getStrainAnalysisDistances,
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

const StrainAnalysis = () => {
  const [availableDistances, setAvailableDistances] = useState([]);
  const [selectedDistance, setSelectedDistance] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const chartRefs = useRef([]);

  useEffect(() => {
    checkBackendStatus();
    loadAvailableDistances();
  }, []);

  const checkBackendStatus = async () => {
    const isHealthy = await checkAPIHealth();
    setApiStatus(isHealthy ? 'connected' : 'disconnected');
  };

  const loadAvailableDistances = async () => {
    try {
      const distances = await getStrainAnalysisDistances();
      setAvailableDistances(distances);
    } catch (error) {
      setError(`Failed to load strain distances: ${error.message}`);
    }
  };

  const handleDistanceSelect = async (distance) => {
    setSelectedDistance(distance);
    setError('');
    setLoading(true);

    try {
      const parsedData = await parseStrainAnalysisFile(distance);
      
      // Filter data for selected distance
      const distanceValue = parseFloat(distance.replace('cm', ''));
      const filteredData = parsedData.data.filter(row => {
        const rowDistance = parseFloat(row['Distance (cm)']);
        return Math.abs(rowDistance - distanceValue) < 0.1; // Allow small floating point differences
      });

      if (filteredData.length === 0) {
        throw new Error(`No data found for distance ${distance}`);
      }

      setAnalysisData({
        ...parsedData,
        data: filteredData,
        distance: distance
      });
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

  const calculateStrain = (data) => {
    // Calculate mechanical strain from force and other parameters
    // This is a simplified calculation - in reality, this would be more complex
    return data.map(point => {
      const force = point['Force (N)'] || 0;
      const area = 100; // mm² - assumed cross-sectional area
      const length = parseFloat(selectedDistance.replace('cm', '')) * 10; // convert to mm
      const youngsModulus = 70000; // MPa - assumed for aluminum
      
      const stress = force / area; // MPa
      const strain = stress / youngsModulus * 1000000; // convert to microstrain
      
      return {
        ...point,
        mechanical_strain: strain
      };
    });
  };

  // Linear regression calculation function
  const calculateLinearRegression = (data) => {
    const n = data.length;
    const sumX = data.reduce((sum, point) => sum + point.x, 0);
    const sumY = data.reduce((sum, point) => sum + point.y, 0);
    const sumXY = data.reduce((sum, point) => sum + (point.x * point.y), 0);
    const sumXX = data.reduce((sum, point) => sum + (point.x * point.x), 0);
    const sumYY = data.reduce((sum, point) => sum + (point.y * point.y), 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R-squared
    const yMean = sumY / n;
    const ssTotal = data.reduce((sum, point) => sum + Math.pow(point.y - yMean, 2), 0);
    const ssResidual = data.reduce((sum, point) => {
      const predicted = slope * point.x + intercept;
      return sum + Math.pow(point.y - predicted, 2);
    }, 0);
    
    const rSquared = 1 - (ssResidual / ssTotal);

    return { slope, intercept, rSquared: Math.max(0, rSquared) }; // Ensure R² is not negative
  };

  const getChannelData = (channel) => {
    if (!analysisData) return null;

    const strainData = calculateStrain(analysisData.data);
    const validData = strainData.filter(point => 
      point[channel] !== null && point[channel] !== undefined && 
      !isNaN(point[channel]) && !isNaN(point.mechanical_strain)
    );

    if (validData.length === 0) return null;

    const scatterData = validData.map(point => ({
      x: point.mechanical_strain,
      y: point[channel]
    }));

    // Calculate linear regression
    const regression = calculateLinearRegression(scatterData);
    
    // Create regression line data points
    const minX = Math.min(...scatterData.map(p => p.x));
    const maxX = Math.max(...scatterData.map(p => p.x));
    const regressionLine = [
      { x: minX, y: regression.slope * minX + regression.intercept },
      { x: maxX, y: regression.slope * maxX + regression.intercept }
    ];

    const baseColor = channel === 'WL_ch1' ? 'rgb(255, 99, 132)' :
                     channel === 'WL_ch2' ? 'rgb(54, 162, 235)' :
                     'rgb(255, 205, 86)';

    return {
      datasets: [
        {
          label: `${channel} Data Points`,
          data: scatterData,
          backgroundColor: baseColor.replace('rgb', 'rgba').replace(')', ', 0.6)'),
          borderColor: baseColor,
          pointRadius: 3,
          pointHoverRadius: 5,
          showLine: false,
          type: 'scatter'
        },
        {
          label: `Regression Line (R² = ${regression.rSquared.toFixed(4)})`,
          data: regressionLine,
          backgroundColor: 'transparent',
          borderColor: baseColor,
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          pointHoverRadius: 0,
          showLine: true,
          type: 'line',
          tension: 0
        }
      ],
      rSquared: regression.rSquared,
      slope: regression.slope,
      intercept: regression.intercept
    };
  };

  const getChartOptions = (channel, regressionData) => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'point',
    },
    plugins: {
      title: {
        display: true,
        text: `Bragg Wavelength vs. Strain - ${channel}`,
        font: {
          size: 16
        }
      },
      legend: {
        display: true,
        position: 'top',
        labels: {
          filter: function(item, chart) {
            // Show both scatter points and regression line in legend
            return true;
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.datasetIndex === 0) {
              // Data points tooltip
              return [
                `Strain: ${context.parsed.x.toFixed(2)} μɛ`,
                `Wavelength: ${context.parsed.y.toFixed(6)} nm`
              ];
            } else {
              // Regression line tooltip
              return [
                `Regression Line`,
                `Strain: ${context.parsed.x.toFixed(2)} μɛ`,
                `Predicted: ${context.parsed.y.toFixed(6)} nm`,
                `R² = ${regressionData?.rSquared.toFixed(4) || 'N/A'}`,
                `Slope: ${regressionData?.slope.toFixed(8) || 'N/A'} nm/μɛ`
              ];
            }
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
          mode: 'xy',
          modifierKey: 'ctrl',
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'xy',
          onZoomComplete: function({chart}) {
            // Sync zoom across all charts
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            const newXMin = xScale.min;
            const newXMax = xScale.max;
            const newYMin = yScale.min;
            const newYMax = yScale.max;
            
            chartRefs.current.forEach(otherChart => {
              if (otherChart && otherChart !== chart) {
                otherChart.zoomScale('x', {min: newXMin, max: newXMax}, 'none');
                otherChart.zoomScale('y', {min: newYMin, max: newYMax}, 'none');
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
          text: 'Mechanical Strain (μɛ)',
          font: {
            size: 14
          }
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Bragg Wavelength Shift (nm)',
          font: {
            size: 14
          }
        },
      },
    },
  });

  const renderChannelChart = (channel) => {
    const data = getChannelData(channel);
    if (!data || !data.datasets[0].data.length) return null;

    const channelNumber = channel.replace('WL_ch', '');
    const regressionData = {
      rSquared: data.rSquared,
      slope: data.slope,
      intercept: data.intercept
    };

    // Determine correlation strength based on R²
    const getCorrelationStrength = (rSquared) => {
      if (rSquared >= 0.9) return { text: 'Very Strong', color: 'text-green-600' };
      if (rSquared >= 0.7) return { text: 'Strong', color: 'text-blue-600' };
      if (rSquared >= 0.5) return { text: 'Moderate', color: 'text-yellow-600' };
      if (rSquared >= 0.3) return { text: 'Weak', color: 'text-orange-600' };
      return { text: 'Very Weak', color: 'text-red-600' };
    };

    const correlation = getCorrelationStrength(regressionData.rSquared);

    return (
      <div key={channel} className="bg-white rounded-lg shadow p-4 mb-6">
        <div style={{ height: '400px' }}>
          <Scatter
            ref={(ref) => {
              if (ref) {
                const index = parseInt(channelNumber) - 1;
                chartRefs.current[index] = ref;
              }
            }}
            data={data}
            options={getChartOptions(`Channel ${channelNumber}`, regressionData)}
          />
        </div>
        
        {/* Regression Statistics */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 bg-gray-50 rounded-lg p-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {regressionData.rSquared.toFixed(4)}
            </div>
            <div className="text-sm text-gray-600">R-squared</div>
            <div className={`text-xs font-medium ${correlation.color}`}>
              {correlation.text} Correlation
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-800">
              {regressionData.slope.toFixed(8)}
            </div>
            <div className="text-sm text-gray-600">Slope (nm/μɛ)</div>
            <div className="text-xs text-gray-500">
              Sensitivity
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-800">
              {regressionData.intercept.toFixed(6)}
            </div>
            <div className="text-sm text-gray-600">Intercept (nm)</div>
            <div className="text-xs text-gray-500">
              Baseline Shift
            </div>
          </div>
        </div>
        
        {/* Regression Equation */}
        <div className="mt-3 text-center">
          <div className="text-sm text-gray-700">
            <strong>Regression Equation:</strong> 
            <span className="font-mono ml-2">
              λ = {regressionData.slope.toFixed(8)} × ε + {regressionData.intercept.toFixed(6)}
            </span>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            where λ is wavelength shift (nm) and ε is mechanical strain (μɛ)
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Strain Analysis Viewer</h2>
      
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

      {/* Distance Selection */}
      {apiStatus === 'connected' && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Select Span Length</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
            {availableDistances.map((distance) => (
              <button
                key={distance}
                onClick={() => handleDistanceSelect(distance)}
                disabled={loading}
                className={`p-4 text-lg border-2 rounded-lg transition-colors ${
                  selectedDistance === distance
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : loading 
                    ? 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed'
                    : 'border-gray-300 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                {distance}
              </button>
            ))}
          </div>
          
          {selectedDistance && (
            <div className="text-sm text-gray-600">
              Selected span length: <span className="font-medium">{selectedDistance}</span>
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
          <p className="mt-2 text-gray-600">Loading strain analysis data...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Data Display */}
      {analysisData && (
        <div className="space-y-6">
          {/* Controls */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-600">
                <strong>Span Length:</strong> {analysisData.distance} | 
                <strong> Data Points:</strong> {analysisData.data.length.toLocaleString()}
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

          {/* Analysis Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h4 className="text-lg font-medium text-blue-800 mb-2">Strain vs Wavelength Analysis</h4>
            <p className="text-blue-700 text-sm mb-3">
              This analysis shows the relationship between mechanical strain and Bragg wavelength shifts 
              for each fiber optic channel at the {analysisData.distance} span length. Each point represents 
              a measurement under different loading conditions with fitted linear regression lines.
            </p>
            
            {/* Summary Statistics */}
            <div className="bg-white rounded-lg p-3 mt-3">
              <h5 className="text-md font-semibold text-gray-800 mb-3">Channel Comparison Summary</h5>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {['WL_ch1', 'WL_ch2', 'WL_ch3'].map(channel => {
                  const data = getChannelData(channel);
                  if (!data || !data.datasets[0].data.length) return null;
                  
                  const channelNum = channel.replace('WL_ch', '');
                  const correlation = data.rSquared >= 0.9 ? 'Very Strong' :
                                    data.rSquared >= 0.7 ? 'Strong' :
                                    data.rSquared >= 0.5 ? 'Moderate' :
                                    data.rSquared >= 0.3 ? 'Weak' : 'Very Weak';
                  
                  const correlationColor = data.rSquared >= 0.9 ? 'text-green-600' :
                                         data.rSquared >= 0.7 ? 'text-blue-600' :
                                         data.rSquared >= 0.5 ? 'text-yellow-600' :
                                         data.rSquared >= 0.3 ? 'text-orange-600' : 'text-red-600';
                  
                  return (
                    <div key={channel} className="text-center border border-gray-200 rounded p-2">
                      <div className="font-medium text-gray-800">Channel {channelNum}</div>
                      <div className="text-lg font-bold text-blue-600">R² = {data.rSquared.toFixed(4)}</div>
                      <div className={`text-xs ${correlationColor}`}>{correlation}</div>
                      <div className="text-xs text-gray-500 mt-1">
                        {data.datasets[0].data.length} points
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="space-y-4">
            {['WL_ch1', 'WL_ch2', 'WL_ch3'].map(channel => renderChannelChart(channel))}
          </div>
        </div>
      )}

      {/* Instructions */}
      {!selectedDistance && apiStatus === 'connected' && (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <h3 className="text-lg font-medium text-gray-800 mb-2">Get Started</h3>
          <p className="text-gray-600 mb-4">
            Select a span length above to view the strain vs wavelength analysis for that configuration.
          </p>
          <div className="text-sm text-gray-500">
            <p>Data is loaded directly from your latest output analysis results.</p>
          </div>
        </div>
      )}

      {selectedDistance && !analysisData && !loading && apiStatus === 'connected' && (
        <div className="bg-gray-50 rounded-lg p-6 text-center">
          <h3 className="text-lg font-medium text-gray-800 mb-2">Loading Analysis Data</h3>
          <p className="text-gray-600 mb-4">
            The analysis data will be loaded to show the relationship between mechanical strain and 
            wavelength shifts for the {selectedDistance} span length.
          </p>
        </div>
      )}
    </div>
  );
};

export default StrainAnalysis;