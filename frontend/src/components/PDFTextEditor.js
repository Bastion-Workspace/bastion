import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import apiService from '../services/apiService';

// PDF.js imports
import * as pdfjsLib from 'pdfjs-dist';
import 'pdfjs-dist/build/pdf.worker.entry';

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const PDFTextEditor = () => {
  const { documentId } = useParams();
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [pdfInfo, setPdfInfo] = useState(null);
  const [selectedPage, setSelectedPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const pdfCanvasRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  
  // Text editing state
  const [textBoxes, setTextBoxes] = useState([]);
  const [activeTextBox, setActiveTextBox] = useState(null);
  
  // Floating controls state
  const [controlsPosition, setControlsPosition] = useState({ x: 20, y: 20 });
  const [isDraggingControls, setIsDraggingControls] = useState(false);
  const [controlsDragStart, setControlsDragStart] = useState({ x: 0, y: 0 });
  
  // PDF.js state
  const [pdfPages, setPdfPages] = useState([]);

  const extractPDFInfo = useCallback(async (documentId) => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('Extracting PDF info for document:', documentId);
      
      const response = await apiService.enhancedExtractPdfInfo({
        document_id: documentId
      });
      
      console.log('PDF info response:', response);
      
      setPdfInfo(response);
      setSelectedPage(0);
      setTextBoxes([]);
      
      // Load the actual PDF for rendering
      await loadPDFDocument(documentId);
      
    } catch (err) {
      console.error('PDF extraction error:', err);
      setError('Failed to extract PDF info: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDocumentById = useCallback(async (documentId) => {
    try {
      setLoading(true);
      console.log('Loading document by ID:', documentId);
      
      const response = await apiService.getDocuments();
      console.log('Documents response:', response);
      
      const document = response.documents?.find(doc => doc.document_id === documentId);
      console.log('Found document:', document);
      
      if (document) {
        setSelectedDocument(document);
        await extractPDFInfo(documentId);
      } else {
        setError(`Document with ID ${documentId} not found`);
      }
    } catch (err) {
      console.error('Document loading error:', err);
      setError('Failed to load document: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [extractPDFInfo]);

  const loadPDFDocument = async (documentId) => {
    try {
      console.log('Loading PDF document for rendering:', documentId);
      
      // Get PDF file URL
      const pdfUrl = apiService.getPdfFileUrl(documentId);
      console.log('PDF URL:', pdfUrl);
      
      // Load PDF with PDF.js
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      
      console.log('PDF loaded successfully:', pdf.numPages, 'pages');
      
      // Load all pages
      const pages = [];
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        pages.push(page);
      }
      setPdfPages(pages);
      
    } catch (err) {
      console.error('PDF loading error:', err);
      setError('Failed to load PDF for rendering: ' + err.message);
    }
  };

  const renderPDFPage = useCallback(async (pageIndex) => {
    if (!pdfPages[pageIndex] || !pdfCanvasRef.current || !pdfInfo) return;
    
    try {
      const page = pdfPages[pageIndex];
      const canvas = pdfCanvasRef.current;
      const context = canvas.getContext('2d');
      
      // Get page info from our extracted data
      const pageInfo = pdfInfo.pages[pageIndex];
      
      // Get the natural viewport (without rotation)
      const viewport = page.getViewport({ scale: 1, rotation: 0 });
      
      // Calculate scale to match our page dimensions
      const scaleX = pageInfo.page_width / viewport.width;
      const scaleY = pageInfo.page_height / viewport.height;
      const scale = Math.min(scaleX, scaleY);
      
      // Create viewport with proper scale and no rotation
      const scaledViewport = page.getViewport({ scale, rotation: 0 });
      
      // Set canvas size to match the scaled viewport
      canvas.width = scaledViewport.width;
      canvas.height = scaledViewport.height;
      canvas.style.width = `${pageInfo.page_width}px`;
      canvas.style.height = `${pageInfo.page_height}px`;
      
      // Clear canvas before rendering
      context.clearRect(0, 0, canvas.width, canvas.height);
      
      // Render PDF page with proper orientation
      const renderContext = {
        canvasContext: context,
        viewport: scaledViewport
      };
      
      await page.render(renderContext).promise;
      console.log(`Rendered PDF page ${pageIndex + 1} with scale ${scale}`);
      
    } catch (err) {
      console.error('PDF page rendering error:', err);
    }
  }, [pdfPages, pdfInfo]);

  // Render PDF page when page changes
  useEffect(() => {
    if (pdfPages.length > 0 && selectedPage >= 0) {
      renderPDFPage(selectedPage);
    }
  }, [pdfPages, selectedPage, renderPDFPage]);

  // Auto-load document if documentId is provided in URL
  useEffect(() => {
    if (documentId) {
      loadDocumentById(documentId);
    }
  }, [documentId, loadDocumentById]);

  const getCanvasCoordinates = (event) => {
    const container = event.currentTarget;
    const rect = container.getBoundingClientRect();
    
    // Account for zoom and pan transformations
    const x = ((event.clientX - rect.left - panOffset.x) / zoomLevel);
    const y = ((event.clientY - rect.top - panOffset.y) / zoomLevel);
    
    return { x, y };
  };

  const handleCanvasClick = (event) => {
    if (!pdfInfo || isPanning) return;
    
    const coords = getCanvasCoordinates(event);
    
    // Check if clicking on existing text box
    const clickedTextBox = textBoxes.find(box => 
      coords.x >= box.x &&
      coords.x <= box.x + box.width &&
      coords.y >= box.y &&
      coords.y <= box.y + box.height
    );
    
    if (clickedTextBox) {
      setActiveTextBox(clickedTextBox.id);
    } else {
      // Create new text box at click position
      const newTextBox = {
        id: Date.now(),
        x: coords.x,
        y: coords.y,
        width: 200,
        height: 30,
        text: 'Click to edit text',
        fontSize: 14,
        color: '#000000',
        backgroundColor: 'rgba(255, 255, 255, 0.8)'
      };
      
      setTextBoxes(prev => [...prev, newTextBox]);
      setActiveTextBox(newTextBox.id);
    }
  };

  const updateTextBox = (id, updates) => {
    setTextBoxes(prev => prev.map(box => 
      box.id === id ? { ...box, ...updates } : box
    ));
  };

  const deleteTextBox = (id) => {
    setTextBoxes(prev => prev.filter(box => box.id !== id));
    setActiveTextBox(null);
  };

  // Floating controls handlers
  const handleControlsMouseDown = useCallback((e) => {
    setIsDraggingControls(true);
    setControlsDragStart({
      x: e.clientX - controlsPosition.x,
      y: e.clientY - controlsPosition.y
    });
    e.preventDefault();
  }, [controlsPosition.x, controlsPosition.y]);

  const handleControlsMouseMove = useCallback((e) => {
    if (isDraggingControls) {
      setControlsPosition({
        x: e.clientX - controlsDragStart.x,
        y: e.clientY - controlsDragStart.y
      });
    }
  }, [isDraggingControls, controlsDragStart.x, controlsDragStart.y]);

  const handleControlsMouseUp = useCallback(() => {
    setIsDraggingControls(false);
  }, []);

  // Add global mouse event listeners for dragging
  useEffect(() => {
    if (isDraggingControls) {
      document.addEventListener('mousemove', handleControlsMouseMove);
      document.addEventListener('mouseup', handleControlsMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleControlsMouseMove);
        document.removeEventListener('mouseup', handleControlsMouseUp);
      };
    }
  }, [isDraggingControls, handleControlsMouseMove, handleControlsMouseUp]);

  const currentPage = pdfInfo?.pages[selectedPage];
  const activeBox = textBoxes.find(box => box.id === activeTextBox);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">PDF Text Editor</h1>
          <p className="text-gray-600">
            Click anywhere on the PDF to add or edit text overlays.
          </p>
        </div>

        {/* Document Info Header */}
        {selectedDocument && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  {selectedDocument.title || selectedDocument.filename}
                </h2>
                <p className="text-sm text-gray-500">{selectedDocument.filename}</p>
              </div>
              <div className="text-sm text-gray-500">
                {pdfInfo ? `${pdfInfo.pages.length} pages` : 'Loading...'}
              </div>
            </div>
          </div>
        )}

        {/* Horizontal Page Navigation */}
        {pdfInfo && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <h3 className="text-lg font-semibold mb-3">Pages</h3>
            <div className="flex space-x-3 overflow-x-auto pb-2">
              {pdfInfo.pages.map((page, index) => (
                <div
                  key={index}
                  className={`flex-shrink-0 p-3 rounded-lg cursor-pointer transition-colors border-2 ${
                    selectedPage === index
                      ? 'bg-blue-100 border-blue-500'
                      : 'bg-gray-50 hover:bg-gray-100 border-gray-200'
                  }`}
                  onClick={() => {
                    setSelectedPage(index);
                    setTextBoxes([]);
                    setActiveTextBox(null);
                  }}
                >
                  <div className="font-medium text-sm text-center">Page {index + 1}</div>
                  <div className="text-xs text-gray-500 text-center mt-1">
                    {Math.round(page.page_width)} × {Math.round(page.page_height)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main Canvas Area */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">
              {currentPage ? `Page ${selectedPage + 1}` : 'Select a page'}
            </h2>
            <div className="text-sm text-gray-500">
              {currentPage && `${textBoxes.length} text overlays`}
            </div>
          </div>

          {currentPage ? (
            <div className="relative border rounded-lg overflow-hidden" style={{ width: '800px', height: '600px', flexShrink: 0 }}>
              <div 
                className="absolute inset-0 overflow-auto bg-gray-100"
                style={{ cursor: isPanning ? 'grabbing' : 'crosshair' }}
                onClick={handleCanvasClick}
                onMouseDown={(e) => {
                  if (e.button === 1 || e.ctrlKey) {
                    setIsPanning(true);
                    setLastPanPoint({ x: e.clientX, y: e.clientY });
                    e.preventDefault();
                  }
                }}
                onMouseMove={(e) => {
                  if (isPanning) {
                    const deltaX = e.clientX - lastPanPoint.x;
                    const deltaY = e.clientY - lastPanPoint.y;
                    setPanOffset(prev => ({
                      x: prev.x + deltaX,
                      y: prev.y + deltaY
                    }));
                    setLastPanPoint({ x: e.clientX, y: e.clientY });
                  }
                }}
                onMouseUp={() => setIsPanning(false)}
                onMouseLeave={() => setIsPanning(false)}
              >
                <div 
                  className="relative bg-white"
                  style={{
                    transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
                    transformOrigin: '0 0',
                    transition: isPanning ? 'none' : 'transform 0.1s ease-out',
                    width: `${currentPage.page_width}px`,
                    height: `${currentPage.page_height}px`,
                    border: '1px solid #ccc'
                  }}
                >
                  {/* PDF rendering canvas */}
                  <canvas
                    ref={pdfCanvasRef}
                    className="absolute inset-0"
                    style={{
                      width: `${currentPage.page_width}px`,
                      height: `${currentPage.page_height}px`,
                      zIndex: 1
                    }}
                  />
                  
                  {/* Text overlays */}
                  {textBoxes.map(box => (
                    <div
                      key={box.id}
                      className={`absolute border-2 ${
                        activeTextBox === box.id ? 'border-blue-500' : 'border-transparent'
                      }`}
                      style={{
                        left: `${box.x}px`,
                        top: `${box.y}px`,
                        width: `${box.width}px`,
                        height: `${box.height}px`,
                        backgroundColor: box.backgroundColor,
                        zIndex: 10,
                        cursor: 'text'
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveTextBox(box.id);
                      }}
                    >
                      <textarea
                        value={box.text}
                        onChange={(e) => updateTextBox(box.id, { text: e.target.value })}
                        className="w-full h-full resize-none border-none outline-none bg-transparent p-1"
                        style={{
                          fontSize: `${box.fontSize}px`,
                          color: box.color,
                          fontFamily: 'Arial, sans-serif'
                        }}
                        placeholder="Enter text..."
                      />
                    </div>
                  ))}
                  
                  {/* Page info overlay */}
                  <div className="absolute top-4 left-4 text-gray-600 text-sm bg-white bg-opacity-90 px-3 py-2 rounded shadow-sm border z-20">
                    <div className="font-medium">PDF Page {selectedPage + 1}</div>
                    <div className="text-xs text-gray-500">
                      {Math.round(currentPage.page_width)} × {Math.round(currentPage.page_height)} pts
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Instructions overlay */}
              <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs p-2 rounded max-w-xs">
                <div>• Click anywhere to add text</div>
                <div>• Click text boxes to edit</div>
                <div>• Ctrl+drag to pan, use controls to zoom</div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg" style={{ height: '600px' }}>
              <div className="text-center">
                <p className="text-gray-500">Select a page to start editing</p>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
            <button 
              onClick={() => setError(null)}
              className="ml-2 text-red-800 hover:text-red-900"
            >
              ×
            </button>
          </div>
        )}

        {/* Floating Controls Panel */}
        {currentPage && (
          <div
            className="fixed bg-white rounded-lg shadow-lg border p-4 z-30"
            style={{
              left: `${controlsPosition.x}px`,
              top: `${controlsPosition.y}px`,
              cursor: isDraggingControls ? 'grabbing' : 'grab',
              minWidth: '300px'
            }}
          >
            {/* Drag Handle */}
            <div
              className="flex items-center justify-between mb-3 pb-2 border-b cursor-grab active:cursor-grabbing"
              onMouseDown={handleControlsMouseDown}
            >
              <h3 className="text-sm font-semibold text-gray-800">Controls</h3>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
              </div>
            </div>

            {/* Zoom Controls */}
            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-700 mb-1">Zoom</label>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setZoomLevel(prev => Math.max(0.25, prev - 0.25))}
                  className="px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded text-xs"
                  disabled={zoomLevel <= 0.25}
                >
                  −
                </button>
                <span className="text-xs font-mono min-w-[50px] text-center">
                  {Math.round(zoomLevel * 100)}%
                </span>
                <button
                  onClick={() => setZoomLevel(prev => Math.min(3, prev + 0.25))}
                  className="px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded text-xs"
                  disabled={zoomLevel >= 3}
                >
                  +
                </button>
                <button
                  onClick={() => {
                    setZoomLevel(1);
                    setPanOffset({ x: 0, y: 0 });
                  }}
                  className="px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-xs"
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Text Box Controls */}
            {activeBox && (
              <>
                <div className="mb-3">
                  <label className="block text-xs font-medium text-gray-700 mb-1">Font Size</label>
                  <input
                    type="range"
                    min="8"
                    max="48"
                    value={activeBox.fontSize}
                    onChange={(e) => updateTextBox(activeBox.id, { fontSize: parseInt(e.target.value) })}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 text-center">{activeBox.fontSize}px</div>
                </div>

                <div className="mb-3">
                  <label className="block text-xs font-medium text-gray-700 mb-1">Text Color</label>
                  <input
                    type="color"
                    value={activeBox.color}
                    onChange={(e) => updateTextBox(activeBox.id, { color: e.target.value })}
                    className="w-full h-8 rounded border"
                  />
                </div>

                <div className="mb-3">
                  <label className="block text-xs font-medium text-gray-700 mb-1">Background</label>
                  <select
                    value={activeBox.backgroundColor}
                    onChange={(e) => updateTextBox(activeBox.id, { backgroundColor: e.target.value })}
                    className="w-full p-1 border border-gray-300 rounded text-xs"
                  >
                    <option value="rgba(255, 255, 255, 0.8)">White (80%)</option>
                    <option value="rgba(255, 255, 255, 1)">White (100%)</option>
                    <option value="rgba(255, 255, 0, 0.8)">Yellow (80%)</option>
                    <option value="rgba(0, 255, 0, 0.8)">Green (80%)</option>
                    <option value="transparent">Transparent</option>
                  </select>
                </div>

                <button
                  onClick={() => deleteTextBox(activeBox.id)}
                  className="w-full px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-xs"
                >
                  Delete Text Box
                </button>
              </>
            )}

            {/* Status */}
            <div className="text-xs text-gray-600 space-y-1 mt-3 pt-3 border-t">
              <div>Text Boxes: {textBoxes.length}</div>
              <div>Active: {activeBox ? 'Yes' : 'None'}</div>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span>Loading PDF...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PDFTextEditor;
