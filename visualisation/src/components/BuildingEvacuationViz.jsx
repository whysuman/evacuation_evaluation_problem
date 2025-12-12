import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, AlertTriangle, Download } from 'lucide-react';

const BuildingEvacuationViz = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [timeStep, setTimeStep] = useState(0);
  const [evacuated, setEvacuated] = useState(0);
  const [scenario, setScenario] = useState('low-density');
  
  // Building layout
  const rooms = [
    { id: 'R1', x: 50, y: 100, people: scenario === 'low-density' ? 10 : 500, name: 'Room 1' },
    { id: 'R2', x: 250, y: 100, people: scenario === 'low-density' ? 5 : 500, name: 'Room 2' },
    { id: 'EXIT', x: 450, y: 100, people: 0, name: 'EXIT', safe: true },
  ];
  
  const hallways = [
    { from: 'R1', to: 'R2', capacity: 100, transit: 1 },
    { from: 'R2', to: 'EXIT', capacity: scenario === 'low-density' ? 100 : 10, transit: 1 },
  ];
  
  const [roomOccupancy, setRoomOccupancy] = useState(
    rooms.reduce((acc, r) => ({ ...acc, [r.id]: r.people }), {})
  );
  
  const totalPeople = rooms.reduce((sum, r) => sum + r.people, 0);
  const bottleneckCapacity = scenario === 'low-density' ? 100 : 10;
  const expectedTime = scenario === 'low-density' ? 2 : Math.ceil(1000 / 10) + 2;
  
  useEffect(() => {
    // Reset when scenario changes
    setTimeStep(0);
    setEvacuated(0);
    setIsRunning(false);
    setRoomOccupancy(rooms.reduce((acc, r) => ({ ...acc, [r.id]: r.people }), {}));
  }, [scenario]);
  
  useEffect(() => {
    if (!isRunning || timeStep >= expectedTime + 5) {
      if (timeStep >= expectedTime + 5) setIsRunning(false);
      return;
    }
    
    const timer = setInterval(() => {
      setTimeStep(t => t + 1);
      
      // Simulate evacuation flow
      setRoomOccupancy(prev => {
        const next = { ...prev };
        
        // Flow from R1 to R2
        const flow1 = Math.min(next['R1'], hallways[0].capacity);
        if (timeStep >= hallways[0].transit) {
          next['R1'] = Math.max(0, next['R1'] - flow1);
          next['R2'] = next['R2'] + flow1;
        }
        
        // Flow from R2 to EXIT (bottleneck!)
        const flow2 = Math.min(next['R2'], bottleneckCapacity);
        if (timeStep >= hallways[1].transit) {
          next['R2'] = Math.max(0, next['R2'] - flow2);
          setEvacuated(e => Math.min(totalPeople, e + flow2));
        }
        
        return next;
      });
    }, 300);
    
    return () => clearInterval(timer);
  }, [isRunning, timeStep, scenario]);
  
  const reset = () => {
    setIsRunning(false);
    setTimeStep(0);
    setEvacuated(0);
    setRoomOccupancy(rooms.reduce((acc, r) => ({ ...acc, [r.id]: r.people }), {}));
  };
  
  const getRoomColor = (room) => {
    if (room.safe) return '#10B981';
    const occupancy = roomOccupancy[room.id];
    const capacity = room.people;
    if (occupancy === 0) return '#9CA3AF';
    if (occupancy / capacity > 0.7) return '#DC2626';
    if (occupancy / capacity > 0.3) return '#F59E0B';
    return '#FCD34D';
  };
  
  const downloadCode = () => {
    // Trigger download of the Python implementation
    const message = `
# The complete Python implementation is available in the artifact above.
# It includes:
# - BuildingEvacuationSystem class
# - Time-expanded graph construction
# - Binary search for minimum evacuation time
# - Three comprehensive experiments
# - Visualization code
# 
# To use: Copy the Python code from the artifact and save as evacuation_system.py
# Then run: python evacuation_system.py
`;
    const blob = new Blob([message], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'README.txt';
    a.click();
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-red-50 to-orange-50 rounded-xl shadow-2xl">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <AlertTriangle className="w-8 h-8 text-red-600" />
          <h1 className="text-3xl font-bold text-gray-800">
            Emergency Building Evacuation
          </h1>
        </div>
        <p className="text-gray-600 ml-11">
          Dynamic Network Flow via Time-Expanded Graph Reduction
        </p>
      </div>

      {/* Theory Box */}
      <div className="bg-blue-50 border-l-4 border-blue-600 p-4 mb-6">
        <h3 className="font-bold text-blue-900 mb-2">Problem Formulation</h3>
        <div className="text-sm text-blue-800 space-y-1">
          <p><strong>Input:</strong> Building graph G = (V,E), capacities c(e), transit times τ(e), initial occupancy S(v)</p>
          <p><strong>Goal:</strong> Find minimum time T such that all people reach safe zones</p>
          <p><strong>Solution:</strong> Binary search on T, construct time-expanded graph G_T with nodes v^t for each location v at time t, solve max-flow</p>
          <p><strong>Key Insight:</strong> Dynamic flow over time ⟺ Static flow in time-expanded graph</p>
        </div>
      </div>

      {/* Scenario Selector */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <h3 className="font-bold text-gray-800 mb-3">Select Scenario</h3>
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setScenario('low-density')}
            className={`p-3 rounded-lg border-2 transition-all ${
              scenario === 'low-density'
                ? 'border-green-500 bg-green-50 shadow-md'
                : 'border-gray-300 hover:border-green-300'
            }`}
          >
            <div className="font-bold text-gray-800">Scenario A: Low Density</div>
            <div className="text-sm text-gray-600 mt-1">
              15 people, wide corridors (capacity 100)
            </div>
            <div className="text-xs text-green-600 mt-1 font-semibold">
              Expected T ≈ 2 (sum of transit times)
            </div>
          </button>
          
          <button
            onClick={() => setScenario('high-density')}
            className={`p-3 rounded-lg border-2 transition-all ${
              scenario === 'high-density'
                ? 'border-red-500 bg-red-50 shadow-md'
                : 'border-gray-300 hover:border-red-300'
            }`}
          >
            <div className="font-bold text-gray-800">Scenario B: High Density</div>
            <div className="text-sm text-gray-600 mt-1">
              1000 people, narrow exit (capacity 10)
            </div>
            <div className="text-xs text-red-600 mt-1 font-semibold">
              Expected T ≈ 102 (people/bottleneck)
            </div>
          </button>
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <svg width="550" height="250" className="border-2 border-gray-300 rounded-lg">
          {/* Hallways */}
          {hallways.map((hall, idx) => {
            const fromRoom = rooms.find(r => r.id === hall.from);
            const toRoom = rooms.find(r => r.id === hall.to);
            const isActive = timeStep > 0;
            
            return (
              <g key={idx}>
                <line
                  x1={fromRoom.x + 40}
                  y1={fromRoom.y + 30}
                  x2={toRoom.x}
                  y2={toRoom.y + 30}
                  stroke={isActive ? '#3B82F6' : '#D1D5DB'}
                  strokeWidth={isActive ? 4 : 2}
                  markerEnd="url(#arrowhead)"
                />
                <text
                  x={(fromRoom.x + toRoom.x) / 2 + 20}
                  y={(fromRoom.y + toRoom.y) / 2 + 15}
                  fontSize="11"
                  fill="#374151"
                  fontWeight="bold"
                >
                  Cap: {hall.capacity}/min
                </text>
                {idx === 1 && scenario === 'high-density' && (
                  <text
                    x={(fromRoom.x + toRoom.x) / 2 + 20}
                    y={(fromRoom.y + toRoom.y) / 2 + 30}
                    fontSize="10"
                    fill="#DC2626"
                    fontWeight="bold"
                  >
                    ⚠️ BOTTLENECK
                  </text>
                )}
              </g>
            );
          })}
          
          {/* Arrow marker */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="3"
              orient="auto"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#3B82F6" />
            </marker>
          </defs>
          
          {/* Rooms */}
          {rooms.map(room => (
            <g key={room.id}>
              <rect
                x={room.x}
                y={room.y}
                width={80}
                height={60}
                rx="8"
                fill={getRoomColor(room)}
                stroke="#1F2937"
                strokeWidth="2"
              />
              <text
                x={room.x + 40}
                y={room.y + 25}
                textAnchor="middle"
                fill="white"
                fontSize="13"
                fontWeight="bold"
              >
                {room.name}
              </text>
              <text
                x={room.x + 40}
                y={room.y + 45}
                textAnchor="middle"
                fill="white"
                fontSize="16"
                fontWeight="bold"
              >
                {room.safe ? evacuated : Math.round(roomOccupancy[room.id])}
              </text>
            </g>
          ))}
          
          {/* Legend */}
          <g transform="translate(20, 190)">
            <rect x="0" y="0" width="15" height="15" fill="#DC2626" />
            <text x="20" y="12" fontSize="11" fill="#374151">High occupancy</text>
            
            <rect x="120" y="0" width="15" height="15" fill="#F59E0B" />
            <text x="140" y="12" fontSize="11" fill="#374151">Medium</text>
            
            <rect x="210" y="0" width="15" height="15" fill="#10B981" />
            <text x="230" y="12" fontSize="11" fill="#374151">Safe zone</text>
          </g>
        </svg>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <div className="bg-white rounded-lg shadow p-3">
          <div className="text-xs text-gray-600">Time Step</div>
          <div className="text-2xl font-bold text-blue-600">{timeStep}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-3">
          <div className="text-xs text-gray-600">Evacuated</div>
          <div className="text-2xl font-bold text-green-600">
            {Math.round(evacuated)} / {totalPeople}
          </div>
        </div>
        <div className="bg-white rounded-lg shadow p-3">
          <div className="text-xs text-gray-600">Expected T</div>
          <div className="text-2xl font-bold text-orange-600">{expectedTime}</div>
        </div>
        <div className="bg-white rounded-lg shadow p-3">
          <div className="text-xs text-gray-600">Progress</div>
          <div className="text-2xl font-bold text-purple-600">
            {Math.round((evacuated / totalPeople) * 100)}%
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <div className="flex gap-3">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
          >
            {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            {isRunning ? 'Pause' : 'Start'}
          </button>
          
          <button
            onClick={reset}
            className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-semibold"
          >
            <RotateCcw className="w-5 h-5" />
            Reset
          </button>
          
          <button
            onClick={downloadCode}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold ml-auto"
          >
            <Download className="w-5 h-5" />
            Download Code
          </button>
        </div>
      </div>

      {/* Algorithm Explanation */}
      <div className="bg-white rounded-lg shadow-md p-5">
        <h3 className="font-bold text-gray-800 mb-3 text-lg">Time-Expanded Graph Construction</h3>
        <div className="space-y-2 text-sm text-gray-700">
          <p><strong>Step 1:</strong> For each time t ∈ [0, T], create nodes v^t (location v at time t)</p>
          <p><strong>Step 2:</strong> Add movement edges (u^t → v^(t+τ)) for each hallway (u,v) with transit time τ</p>
          <p><strong>Step 3:</strong> Add holdover edges (v^t → v^(t+1)) to allow waiting in place</p>
          <p><strong>Step 4:</strong> Connect super-source to initial occupancy, safe zones to super-sink</p>
          <p><strong>Step 5:</strong> Run max-flow. If flow = total people, then T is feasible!</p>
          <p className="pt-2 border-t border-gray-200 mt-3">
            <strong>Binary Search:</strong> Find minimum T by testing: if T works, try T/2; if not, try 2T
          </p>
        </div>
      </div>
    </div>
  );
};

export default BuildingEvacuationViz;