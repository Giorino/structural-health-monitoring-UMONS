import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import InterrogatorData from './components/InterrogatorData';
import OutputData from './components/OutputData';
import StrainAnalysis from './components/StrainAnalysis';

function Navigation() {
  const location = useLocation();
  
  const isActive = (path) => location.pathname === path;
  
  return (
    <nav className="bg-blue-600 text-white p-4 shadow-lg">
      <div className="container mx-auto">
        <h1 className="text-2xl font-bold mb-4">Structural Health Monitoring Dashboard</h1>
        <div className="flex space-x-4">
          <Link
            to="/"
            className={`px-4 py-2 rounded transition-colors ${
              isActive('/') ? 'bg-blue-800' : 'hover:bg-blue-700'
            }`}
          >
            Interrogator Data
          </Link>
          <Link
            to="/output"
            className={`px-4 py-2 rounded transition-colors ${
              isActive('/output') ? 'bg-blue-800' : 'hover:bg-blue-700'
            }`}
          >
            Output Data
          </Link>
          <Link
            to="/strain"
            className={`px-4 py-2 rounded transition-colors ${
              isActive('/strain') ? 'bg-blue-800' : 'hover:bg-blue-700'
            }`}
          >
            Strain Analysis
          </Link>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navigation />
        <div className="container mx-auto py-6">
          <Routes>
            <Route path="/" element={<InterrogatorData />} />
            <Route path="/output" element={<OutputData />} />
            <Route path="/strain" element={<StrainAnalysis />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;