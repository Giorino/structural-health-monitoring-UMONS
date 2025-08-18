import Papa from 'papaparse';

const API_BASE_URL = '/api';

// Helper function to parse text data (for interrogator files)
const parseTextData = (content, filename) => {
  return new Promise((resolve, reject) => {
    Papa.parse(content, {
      complete: (results) => {
        try {
          const data = results.data;
          if (data.length < 2) {
            reject(new Error('File appears to be empty or corrupted'));
            return;
          }

          // Skip first row (header) and parse data
          const parsedData = data.slice(1).map((row, index) => {
            if (row.length < 3) return null; // Skip incomplete rows
            
            return {
              timestamp: row[0],
              time: parseFloat(row[1]),
              wl1: parseFloat(row[2]),
              wl2: parseFloat(row[3]) || null,
              wl3: parseFloat(row[4]) || null
            };
          }).filter(row => row && !isNaN(row.time)); // Filter out null and invalid rows

          resolve({
            filename,
            data: parsedData,
            channels: getChannelCount(parsedData[0])
          });
        } catch (error) {
          reject(error);
        }
      },
      error: (error) => {
        reject(error);
      },
      header: false,
      delimiter: '\t', // Tab-separated values
      skipEmptyLines: true
    });
  });
};

// Helper function to parse CSV data (for output files)
const parseCSVData = (content, filename) => {
  return new Promise((resolve, reject) => {
    Papa.parse(content, {
      complete: (results) => {
        try {
          const data = results.data;
          if (data.length < 2) {
            reject(new Error('File appears to be empty or corrupted'));
            return;
          }

          // Parse CSV data
          const headers = data[0];
          const parsedData = data.slice(1).map((row) => {
            const rowData = {};
            headers.forEach((header, index) => {
              const value = row[index];
              // Try to parse as number, otherwise keep as string
              rowData[header] = isNaN(parseFloat(value)) ? value : parseFloat(value);
            });
            return rowData;
          }).filter(row => Object.keys(row).length > 0);

          resolve({
            filename,
            data: parsedData,
            headers: headers
          });
        } catch (error) {
          reject(error);
        }
      },
      error: (error) => {
        reject(error);
      },
      header: false,
      skipEmptyLines: true
    });
  });
};

// API Functions

export const getInterrogatorFileList = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/interrogator-files`);
    if (!response.ok) {
      throw new Error(`Failed to fetch interrogator files: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching interrogator files:', error);
    return [];
  }
};

export const parseInterrogatorFile = async (filename) => {
  try {
    const response = await fetch(`${API_BASE_URL}/interrogator-files/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch interrogator file: ${response.statusText}`);
    }
    const { content } = await response.json();
    return await parseTextData(content, filename);
  } catch (error) {
    throw new Error(`Error loading interrogator file: ${error.message}`);
  }
};

export const getOutputDateFolders = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/output-dates`);
    if (!response.ok) {
      throw new Error(`Failed to fetch output dates: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching output dates:', error);
    return [];
  }
};

export const getOutputFiles = async (date) => {
  try {
    const response = await fetch(`${API_BASE_URL}/output-files/${date}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch output files: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching output files:', error);
    return [];
  }
};

export const parseOutputFile = async (date, config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/output-data/${date}/${config}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch output data: ${response.statusText}`);
    }
    const { content, filename } = await response.json();
    return await parseCSVData(content, filename);
  } catch (error) {
    throw new Error(`Error loading output file: ${error.message}`);
  }
};

export const getStrainAnalysisDistances = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/strain-distances`);
    if (!response.ok) {
      throw new Error(`Failed to fetch strain distances: ${response.statusText}`);
    }
    const { distances } = await response.json();
    return distances;
  } catch (error) {
    console.error('Error fetching strain distances:', error);
    return ['11.0cm', '15.0cm', '19.0cm', '23.0cm', '27.0cm'];
  }
};

export const parseStrainAnalysisFile = async (distance) => {
  try {
    const response = await fetch(`${API_BASE_URL}/strain-analysis/${distance}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch strain analysis data: ${response.statusText}`);
    }
    const { content, filename } = await response.json();
    return await parseCSVData(content, filename);
  } catch (error) {
    throw new Error(`Error loading strain analysis data: ${error.message}`);
  }
};

// Helper functions
const getChannelCount = (firstRow) => {
  let channels = 1; // At least WL1
  if (firstRow.wl2 !== null) channels++;
  if (firstRow.wl3 !== null) channels++;
  return channels;
};

export const formatTime = (timeInSeconds) => {
  const hours = Math.floor(timeInSeconds / 3600);
  const minutes = Math.floor((timeInSeconds % 3600) / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  const milliseconds = Math.floor((timeInSeconds % 1) * 1000);
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
};

// Health check function
export const checkAPIHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
};