import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import apiService from '../services/apiService';

// PDF.js imports
import * as pdfjsLib from 'pdfjs-dist';
import 'pdfjs-dist/build/pdf.worker.entry';

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const EnhancedPDFSegmentationPage = () => {
  const { documentId } = useParams();
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [pdfInfo, setPdfInfo] = useState(null);
  const [selectedPage, setSelectedPage] = useState(0);
  const [segments, setSegments] = useState([]);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingBounds, setDrawingBounds] = useState(null);
  const [multiSelectMode, setMultiSelectMode] = useState(false);
  const [multiSelections, setMultiSelections] = useState([]);
  const [segmentType, setSegmentType] = useState('article');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [editingText, setEditingText] = useState('');
  const [showTextEditor, setShowTextEditor] = useState(false);
  
  const canvasRef = useRef(null);
  const pdfCanvasRef = useRef(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  
  // Floating controls state
  const [controlsPosition, setControlsPosition] = useState({ x: 20, y: 20 });
  const [isDraggingControls, setIsDraggingControls] = useState(false);
  const [controlsDragStart, setControlsDragStart] = useState({ x: 0, y: 0 });
  
  // PDF.js state
  const [pdfPages, setPdfPages] = useState([]);

  const segmentTypes = [
    'article', 'advertisement', 'image', 'caption', 'headline', 
    'byline', 'dateline', 'masthead', 'classified', 'editorial',
    'letter', 'cartoon', 'chart', 'table', 'footer', 'header', 'other'
  ];

  const segmentColors = useMemo(() => ({
    article: '#3B82F6',
    advertisement: '#EF4444', 
    image: '#10B981',
    caption: '#F59E0B',
    headline: '#8B5CF6',
    byline: '#06B6D4',
    dateline: '#84CC16',
    masthead: '#F97316',
    classified: '#EC4899',
    editorial: '#6366F1',
    letter: '#14B8A6',
    cartoon: '#F472B6',
    chart: '#A855F7',
    table: '#22D3EE',
    footer: '#94A3B8',
    header: '#64748B',
    other: '#6B7280'
  }), []);

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
      setSegments([]);
      setSelectedSegment(null);
      
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

  const loadPageSegments = useCallback(async () => {
    if (!pdfInfo || selectedPage < 0) return;
    
    try {
      // For now, we'll store segments in component state
      // In a real implementation, you'd load from the backend
      setSegments([]);
    } catch (err) {
      setError('Failed to load page segments: ' + err.message);
    }
  }, [pdfInfo, selectedPage]);

  // Auto-load document if documentId is provided in URL
  useEffect(() => {
    if (documentId) {
      loadDocumentById(documentId);
    }
  }, [documentId, loadDocumentById]);

  useEffect(() => {
    if (pdfInfo) {
      loadPageSegments();
    }
  }, [pdfInfo, selectedPage, loadPageSegments]);

  const drawSegments = useCallback(() => {
    if (!canvasRef.current || !pdfInfo) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw existing segments
    segments.forEach(segment => {
      const color = segmentColors[segment.segment_type] || '#6B7280';
      
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '20'; // 20% opacity
      ctx.lineWidth = 2;
      
      ctx.fillRect(
        segment.bounds.x,
        segment.bounds.y,
        segment.bounds.width,
        segment.bounds.height
      );
      
      ctx.strokeRect(
        segment.bounds.x,
        segment.bounds.y,
        segment.bounds.width,
        segment.bounds.height
      );
      
      // Draw label
      ctx.fillStyle = color;
      ctx.font = '12px Arial';
      ctx.fillText(
        segment.segment_type,
        segment.bounds.x + 5,
        segment.bounds.y + 15
      );
    });
    
    // Draw multi-selections
    if (multiSelectMode) {
      multiSelections.forEach((selection, index) => {
        const color = segmentColors[segmentType] || '#6B7280';
        ctx.strokeStyle = color;
        ctx.fillStyle = color + '10';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        
        ctx.fillRect(
          selection.x,
          selection.y,
          selection.width,
          selection.height
        );
        
        ctx.strokeRect(
          selection.x,
          selection.y,
          selection.width,
          selection.height
        );
        
        // Draw selection number
        ctx.fillStyle = color;
        ctx.font = 'bold 14px Arial';
        ctx.fillText(
          (index + 1).toString(),
          selection.x + 5,
          selection.y + 20
        );
        
        ctx.setLineDash([]);
      });
    }
    
    // Draw current drawing bounds
    if (drawingBounds) {
      const color = segmentColors[segmentType] || '#6B7280';
      ctx.strokeStyle = color;
      ctx.fillStyle = color + '20';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      ctx.fillRect(
        drawingBounds.x,
        drawingBounds.y,
        drawingBounds.width,
        drawingBounds.height
      );
      
      ctx.strokeRect(
        drawingBounds.x,
        drawingBounds.y,
        drawingBounds.width,
        drawingBounds.height
      );
      
      ctx.setLineDash([]);
    }
  }, [segments, pdfInfo, drawingBounds, segmentType, segmentColors, multiSelectMode, multiSelections]);

  useEffect(() => {
    if (pdfInfo) {
      drawSegments();
    }
  }, [segments, pdfInfo, drawSegments]);

  const setupCanvas = useCallback(() => {
    if (!canvasRef.current || !pdfInfo) return;
    
    const canvas = canvasRef.current;
    const currentPage = pdfInfo.pages[selectedPage];
    
    if (currentPage) {
      // Set canvas size to match PDF page dimensions
      canvas.width = currentPage.page_width;
      canvas.height = currentPage.page_height;
      
      // Set canvas display size
      canvas.style.width = `${currentPage.page_width}px`;
      canvas.style.height = `${currentPage.page_height}px`;
      
      drawSegments();
    }
  }, [pdfInfo, selectedPage, drawSegments]);

  useEffect(() => {
    setupCanvas();
  }, [setupCanvas]);

  const getCanvasCoordinates = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Account for zoom and pan transformations
    const x = ((event.clientX - rect.left - panOffset.x) / zoomLevel);
    const y = ((event.clientY - rect.top - panOffset.y) / zoomLevel);
    
    return { x, y };
  };

  const handleMouseDown = (event) => {
    if (!pdfInfo) return;
    
    const coords = getCanvasCoordinates(event);
    
    // Check if clicking on existing segment
    const clickedSegment = segments.find(segment => 
      coords.x >= segment.bounds.x &&
      coords.x <= segment.bounds.x + segment.bounds.width &&
      coords.y >= segment.bounds.y &&
      coords.y <= segment.bounds.y + segment.bounds.height
    );
    
    if (clickedSegment) {
      setSelectedSegment(clickedSegment);
      setEditingText(clickedSegment.text || '');
      setShowTextEditor(true);
    } else {
      // Start drawing new segment
      setIsDrawing(true);
      setSelectedSegment(null);
      setDrawingBounds({
        x: coords.x,
        y: coords.y,
        width: 0,
        height: 0
      });
    }
  };

  const handleMouseMove = (event) => {
    if (!isDrawing || !drawingBounds) return;
    
    const coords = getCanvasCoordinates(event);
    
    setDrawingBounds(prev => ({
      ...prev,
      width: coords.x - prev.x,
      height: coords.y - prev.y
    }));
    
    drawSegments();
  };

  const handleMouseUp = () => {
    if (isDrawing && drawingBounds) {
      // Normalize bounds (handle negative width/height)
      const normalizedBounds = {
        x: drawingBounds.width < 0 ? drawingBounds.x + drawingBounds.width : drawingBounds.x,
        y: drawingBounds.height < 0 ? drawingBounds.y + drawingBounds.height : drawingBounds.y,
        width: Math.abs(drawingBounds.width),
        height: Math.abs(drawingBounds.height)
      };
      
      // Only create segment if bounds are large enough
      if (normalizedBounds.width > 10 && normalizedBounds.height > 10) {
        if (multiSelectMode) {
          setMultiSelections(prev => [...prev, normalizedBounds]);
        } else {
          createSingleSegment(normalizedBounds);
        }
      }
    }
    
    setIsDrawing(false);
    setDrawingBounds(null);
  };

  const createSingleSegment = async (bounds) => {
    try {
      setLoading(true);
      
      // First extract text from the region
      const textResponse = await apiService.extractTextFromPdfRegion({
        document_id: selectedDocument.document_id,
        page_number: selectedPage,
        bounds: bounds
      });
      
      setExtractedText(textResponse.extracted_text || '');
      setEditingText(textResponse.extracted_text || '');
      
      // Create the segment with extracted text
      const segmentResponse = await apiService.createEnhancedSegment({
        document_id: selectedDocument.document_id,
        page_number: selectedPage,
        segment_type: segmentType,
        bounds: bounds,
        text: textResponse.extracted_text || ''
      });
      
      setSegments(prev => [...prev, {
        ...segmentResponse,
        bounds: bounds,
        segment_type: segmentType,
        text: textResponse.extracted_text || ''
      }]);
      
      setShowTextEditor(true);
      
    } catch (err) {
      setError('Failed to create segment: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const createMultipleSegments = async () => {
    if (multiSelections.length === 0) return;
    
    try {
      setLoading(true);
      
      const response = await apiService.createMultipleSelections({
        document_id: selectedDocument.document_id,
        page_number: selectedPage,
        segment_type: segmentType,
        selections: multiSelections
      });
      
      setSegments(prev => [...prev, ...response.segments]);
      setMultiSelections([]);
      setMultiSelectMode(false);
      
    } catch (err) {
      setError('Failed to create multiple segments: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const updateSegmentText = async () => {
    if (!selectedSegment) return;
    
    try {
      setLoading(true);
      
      await apiService.editSegmentText(selectedSegment.segment_id, {
        text: editingText
      });
      
      setSegments(prev => prev.map(s => 
        s.segment_id === selectedSegment.segment_id 
          ? { ...s, text: editingText }
          : s
      ));
      
      setSelectedSegment(prev => ({ ...prev, text: editingText }));
      setShowTextEditor(false);
      
    } catch (err) {
      setError('Failed to update segment text: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const cropSegmentToPdf = async (segment) => {
    try {
      setLoading(true);
      
      const response = await apiService.cropSegmentToPdf(segment.segment_id, {
        output_filename: `${selectedDocument.filename}_segment_${segment.segment_id}.pdf`
      });
      
      // Download the cropped PDF
      const blob = await apiService.downloadCroppedPdf(response.output_filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = response.output_filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (err) {
      setError('Failed to crop segment to PDF: ' + err.message);
    } finally {
      setLoading(false);
    }
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

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Enhanced PDF Segmentation</h1>
          <p className="text-gray-600">
            Select portions of PDF pages, extract and edit text, and create standalone PDF documents.
          </p>
        </div>

        {/* Debug Information */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <h3 className="text-sm font-semibold text-yellow-800 mb-2">Debug Information</h3>
          <div className="text-xs text-yellow-700 space-y-1">
            <div>Document ID from URL: {documentId || 'None'}</div>
            <div>Selected Document: {selectedDocument ? selectedDocument.document_id : 'None'}</div>
            <div>PDF Info: {pdfInfo ? `${pdfInfo.pages?.length || 0} pages` : 'None'}</div>
            <div>Loading: {loading ? 'Yes' : 'No'}</div>
            <div>Error: {error || 'None'}</div>
          </div>
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
                  onClick={() => setSelectedPage(index)}
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
                {currentPage && `${segments.length} segments`}
              </div>
            </div>

            {currentPage ? (
              <div className="relative border rounded-lg overflow-hidden" style={{ width: '800px', height: '600px', flexShrink: 0 }}>
                <div 
                  className="absolute inset-0 overflow-auto bg-gray-100"
                  style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
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
                    {/* PDF rendering canvas - background layer */}
                    <canvas
                      ref={pdfCanvasRef}
                      className="absolute inset-0 bg-white"
                      style={{
                        width: `${currentPage.page_width}px`,
                        height: `${currentPage.page_height}px`,
                        zIndex: 1
                      }}
                    />

                    {/* Drawing canvas - interaction layer above PDF */}
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 pointer-events-auto"
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      style={{ 
                        cursor: isDrawing ? 'crosshair' : 'default',
                        zIndex: 2,
                        backgroundColor: 'transparent'
                      }}
                    />
                    
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
                  <div>• Click and drag to select regions</div>
                  <div>• Click segments to edit text</div>
                  <div>• Use multi-select for batch operations</div>
                  <div>• Ctrl+drag to pan, scroll to zoom</div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg" style={{ height: '600px' }}>
                <div className="text-center">
                  <p className="text-gray-500">Select a page to start segmentation</p>
                </div>
              </div>
            )}

        </div>

        {/* Text Editor Modal */}
        {showTextEditor && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
              <h3 className="text-lg font-semibold mb-4">Edit Segment Text</h3>
              
              {extractedText && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Extracted Text:
                  </label>
                  <div className="p-3 bg-gray-50 rounded border text-sm">
                    {extractedText}
                  </div>
                </div>
              )}
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Edit Text:
                </label>
                <textarea
                  value={editingText}
                  onChange={(e) => setEditingText(e.target.value)}
                  placeholder="Enter or correct the text content..."
                  className="w-full p-3 border border-gray-300 rounded-md h-40 resize-none"
                />
              </div>
              
              <div className="flex justify-between">
                <div className="flex space-x-2">
                  {selectedSegment && (
                    <button
                      onClick={() => cropSegmentToPdf(selectedSegment)}
                      className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                    >
                      Download as PDF
                    </button>
                  )}
                </div>
                
                <div className="flex space-x-2">
                  <button
                    onClick={() => setShowTextEditor(false)}
                    className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={updateSegmentText}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  >
                    Save Text
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

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

            {/* Segment Type */}
            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Segment Type
              </label>
              <select
                value={segmentType}
                onChange={(e) => setSegmentType(e.target.value)}
                className="w-full p-1 border border-gray-300 rounded text-xs"
              >
                {segmentTypes.map(type => (
                  <option key={type} value={type}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* Multi-Select Mode */}
            <div className="mb-3">
              <button
                onClick={() => {
                  setMultiSelectMode(!multiSelectMode);
                  setMultiSelections([]);
                }}
                className={`w-full px-2 py-1 rounded text-xs ${
                  multiSelectMode 
                    ? 'bg-green-500 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300'
                }`}
              >
                Multi-Select {multiSelectMode ? 'ON' : 'OFF'}
              </button>
            </div>

            {/* Multi-Select Actions */}
            {multiSelectMode && (
              <div className="mb-3">
                <div className="flex space-x-1">
                  {multiSelections.length > 0 && (
                    <button
                      onClick={createMultipleSegments}
                      className="flex-1 px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-xs"
                    >
                      Create {multiSelections.length}
                    </button>
                  )}
                  <button
                    onClick={() => setMultiSelections([])}
                    className="flex-1 px-2 py-1 bg-gray-600 text-white rounded hover:bg-gray-700 text-xs"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}

            {/* Status */}
            <div className="text-xs text-gray-600 space-y-1">
              <div>Segments: {segments.length}</div>
              <div>Selections: {multiSelections.length}</div>
              <div>Mode: {multiSelectMode ? 'Multi-Select' : 'Single'}</div>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span>Processing...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedPDFSegmentationPage;
