#!/bin/bash

echo "🔬 Starting Structural Health Monitoring Dashboard..."
echo "======================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "webapp" ]; then
    echo "❌ webapp directory not found. Please run this script from the project root."
    exit 1
fi

# Navigate to webapp directory
cd webapp

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Function to clean up background processes on exit
cleanup() {
    echo "🧹 Shutting down servers..."
    kill $SERVER_PID
    exit
}

trap cleanup SIGINT SIGTERM

# Check for --share flag
if [ "$1" == "--share" ]; then
  echo "🚀 Starting servers and ngrok tunnel..."
  
  # Start all services concurrently in the background
  npm run dev:share &
  SERVER_PID=$!
  
  echo "⏳ Waiting for ngrok to establish tunnel..."
  sleep 5 # Give ngrok a few seconds to start

  # Fetch the public URL from the ngrok client API
  NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | jq -r ".tunnels[0].public_url")

  if [ -z "$NGROK_URL" ] || [ "$NGROK_URL" == "null" ]; then
    echo "❌ Could not retrieve ngrok URL. Please check if ngrok is running correctly."
  else
    echo ""
    echo "✅ Public URL is ready!"
    echo "=========================================="
    echo "🔗 URL: $NGROK_URL"
    echo "=========================================="
    echo ""
    echo "This URL has been saved to webapp/ngrok_url.log"
    echo "$NGROK_URL" > ngrok_url.log
  fi

  echo "➡️ All services are running in the background."
  echo "   Press Ctrl+C in this terminal to shut everything down."
  wait $SERVER_PID

else
  # Start just the local servers
  echo "🚀 Starting backend and frontend servers..."
  echo ""
  echo "Frontend: http://localhost:5173"
  echo "Backend API: http://localhost:3001"
  echo "Press Ctrl+C to stop both servers"
  echo ""
  npm run dev:full
fi