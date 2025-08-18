import express from 'express';
import cors from 'cors';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

// Enable CORS for development
app.use(cors());
app.use(express.json());

// Base paths to data directories
const BASE_PATH = path.join(__dirname, '..');
const INTERROGATOR_PATH = path.join(BASE_PATH, 'interrogator-data');
const OUTPUT_PATH = path.join(BASE_PATH, 'output');
const STRAIN_PATH = path.join(BASE_PATH, 'strain_wavelength_analysis_plots');

// API Routes

// Get list of interrogator files
app.get('/api/interrogator-files', async (req, res) => {
  try {
    const files = await fs.readdir(INTERROGATOR_PATH);
    const txtFiles = files
      .filter(file => file.endsWith('.txt') && file.includes('interrogator'))
      .sort();
    res.json(txtFiles);
  } catch (error) {
    console.error('Error reading interrogator files:', error);
    res.status(500).json({ error: 'Failed to read interrogator files' });
  }
});

// Get specific interrogator file data
app.get('/api/interrogator-files/:filename', async (req, res) => {
  try {
    const { filename } = req.params;
    const filePath = path.join(INTERROGATOR_PATH, filename);
    
    // Check if file exists
    if (!await fs.pathExists(filePath)) {
      return res.status(404).json({ error: 'File not found' });
    }

    const content = await fs.readFile(filePath, 'utf-8');
    res.json({ filename, content });
  } catch (error) {
    console.error('Error reading interrogator file:', error);
    res.status(500).json({ error: 'Failed to read interrogator file' });
  }
});

// Get list of output date folders
app.get('/api/output-dates', async (req, res) => {
  try {
    const items = await fs.readdir(OUTPUT_PATH);
    const folders = [];
    
    for (const item of items) {
      const itemPath = path.join(OUTPUT_PATH, item);
      const stats = await fs.stat(itemPath);
      if (stats.isDirectory() && /^\d{8}_\d{6}$/.test(item)) {
        folders.push(item);
      }
    }
    
    res.json(folders.sort().reverse()); // Most recent first
  } catch (error) {
    console.error('Error reading output dates:', error);
    res.status(500).json({ error: 'Failed to read output dates' });
  }
});

// Get list of files for a specific date
app.get('/api/output-files/:date', async (req, res) => {
  try {
    const { date } = req.params;
    const datePath = path.join(OUTPUT_PATH, date);
    
    if (!await fs.pathExists(datePath)) {
      return res.status(404).json({ error: 'Date folder not found' });
    }

    const files = await fs.readdir(datePath);
    const csvFiles = files
      .filter(file => file.endsWith('.csv') && file.startsWith('merged_'))
      .map(file => {
        // Extract configuration name from filename
        // e.g., "merged_11cm-12layers-1_20250818_1031.csv" -> "11cm-12layers-1"
        const match = file.match(/^merged_(.+?)_\d{8}_\d{4}\.csv$/);
        return match ? match[1] : file;
      })
      .sort();
    
    res.json(csvFiles);
  } catch (error) {
    console.error('Error reading output files:', error);
    res.status(500).json({ error: 'Failed to read output files' });
  }
});

// Get combined output data for a specific date
app.get('/api/output-data/combined/:date', async (req, res) => {
  try {
    const { date } = req.params;
    const datePath = path.join(OUTPUT_PATH, date);

    if (!await fs.pathExists(datePath)) {
      return res.status(404).json({ error: 'Date folder not found' });
    }

    const files = await fs.readdir(datePath);
    const csvFiles = files
      .filter(file => file.endsWith('.csv') && file.startsWith('merged_'))
      .sort();

    let combinedData = [];
    let fileSegments = [];
    let currentIndex = 0;

    for (const file of csvFiles) {
      const filePath = path.join(datePath, file);
      const content = await fs.readFile(filePath, 'utf-8');
      const parsed = await new Promise((resolve, reject) => {
        // Using PapaParse on the server
        import('papaparse').then(Papa => {
          Papa.default.parse(content, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: results => resolve(results.data),
            error: err => reject(err),
          });
        });
      });

      const configName = file.match(/^merged_(.+?)_\d{8}_\d{4}\.csv$/)[1];
      parsed.forEach(row => {
        combinedData.push({ ...row, source_file: configName });
      });
      
      fileSegments.push({
        name: configName,
        start: currentIndex,
        end: currentIndex + parsed.length -1,
      });
      currentIndex += parsed.length;
    }
    
    res.json({ combinedData, fileSegments });

  } catch (error) {
    console.error('Error reading combined output data:', error);
    res.status(500).json({ error: 'Failed to read combined output data' });
  }
});

// Get specific output file data
app.get('/api/output-data/:date/:config', async (req, res) => {
  try {
    const { date, config } = req.params;
    const datePath = path.join(OUTPUT_PATH, date);
    
    // Find the actual filename
    const files = await fs.readdir(datePath);
    const targetFile = files.find(file => 
      file.startsWith(`merged_${config}_`) && file.endsWith('.csv')
    );
    
    if (!targetFile) {
      return res.status(404).json({ error: 'Output file not found' });
    }

    const filePath = path.join(datePath, targetFile);
    const content = await fs.readFile(filePath, 'utf-8');
    
    res.json({ 
      filename: targetFile, 
      content,
      date,
      config 
    });
  } catch (error) {
    console.error('Error reading output file:', error);
    res.status(500).json({ error: 'Failed to read output file' });
  }
});

// Get strain analysis distances
app.get('/api/strain-distances', async (req, res) => {
  try {
    // Use the latest output folder to get strain analysis data
    const outputDates = await fs.readdir(OUTPUT_PATH);
    const latestDate = outputDates
      .filter(item => /^\d{8}_\d{6}$/.test(item))
      .sort()
      .reverse()[0];
    
    if (!latestDate) {
      return res.status(404).json({ error: 'No output data found' });
    }

    // Return standard distances based on your data structure
    const distances = ['11.0cm', '15.0cm', '19.0cm', '23.0cm', '27.0cm'];
    res.json({ distances, latestDate });
  } catch (error) {
    console.error('Error reading strain distances:', error);
    res.status(500).json({ error: 'Failed to read strain distances' });
  }
});

// Get strain analysis data for a specific distance
app.get('/api/strain-analysis/:distance', async (req, res) => {
  try {
    const { distance } = req.params;
    
    // Get the latest output folder
    const outputDates = await fs.readdir(OUTPUT_PATH);
    const latestDate = outputDates
      .filter(item => /^\d{8}_\d{6}$/.test(item))
      .sort()
      .reverse()[0];
    
    if (!latestDate) {
      return res.status(404).json({ error: 'No output data found' });
    }

    // Look for strain analysis results file
    const latestPath = path.join(OUTPUT_PATH, latestDate);
    const strainFile = 'strain_analysis_results.csv';
    const strainFilePath = path.join(latestPath, strainFile);
    
    if (!await fs.pathExists(strainFilePath)) {
      return res.status(404).json({ error: 'Strain analysis file not found' });
    }

    const content = await fs.readFile(strainFilePath, 'utf-8');
    
    res.json({ 
      filename: strainFile,
      content,
      distance,
      date: latestDate
    });
  } catch (error) {
    console.error('Error reading strain analysis:', error);
    res.status(500).json({ error: 'Failed to read strain analysis data' });
  }
});


// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Structural Health Monitoring API is running',
    paths: {
      interrogator: INTERROGATOR_PATH,
      output: OUTPUT_PATH,
      strain: STRAIN_PATH
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Structural Health Monitoring API server running on http://localhost:${PORT}`);
  console.log(`📁 Interrogator data path: ${INTERROGATOR_PATH}`);
  console.log(`📁 Output data path: ${OUTPUT_PATH}`);
  console.log(`📁 Strain analysis path: ${STRAIN_PATH}`);
});
