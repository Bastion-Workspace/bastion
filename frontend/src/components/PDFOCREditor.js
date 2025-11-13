import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import apiService from '../services/apiService';

// PDF.js imports
import * as pdfjsLib from 'pdfjs-dist';
import 'pdfjs-dist/build/pdf.worker.entry';

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const PDFOCREditor = () => {
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
  
  // OCR text editing state
  const [ocrTextBlocks, setOcrTextBlocks] = useState([]);
  const [activeTextBlock, setActiveTextBlock] = useState(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Floating controls state
  const [controlsPosition, setControlsPosition] = useState({ x: 20, y: 20 });
  const [isDraggingControls, setIsDraggingControls] = useState(false);
  const [controlsDragStart, setControlsDragStart] = useState({ x: 0, y: 0 });
  
  // PDF.js state
  const [pdfPages, setPdfPages] = useState([]);

  const loadOCRTextForPage = useCallback(async (documentId, pageIndex) => {
    try {
      console.log('Loading OCR text for page:', pageIndex);
      
      // Get the page info from our extracted data
      const pageInfo = pdfInfo?.pages[pageIndex];
      if (!pageInfo) return;
      
      // Convert the text blocks to editable format
      const textBlocks = pageInfo.text_blocks?.map((block, index) => ({
        id: `${pageIndex}-${index}`,
        pageIndex,
        originalText: block.text,
        editedText: block.text,
        bbox: block.bbox,
        confidence: block.confidence || 0.9,
        isEdited: false
      })) || [];
      
      setOcrTextBlocks(textBlocks);
      setActiveTextBlock(null);
      setHasUnsavedChanges(false);
      
      console.log('Loaded OCR text blocks:', textBlocks.length);
      
    } catch (err) {
      console.error('OCR text loading error:', err);
      setError('Failed to load OCR text: ' + err.message);
    }
  }, [pdfInfo]);

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
      
      // Load the actual PDF for rendering
      await loadPDFDocument(documentId);
      
      // Load OCR text for the first page
      if (response.pages && response.pages.length > 0) {
        await loadOCRTextForPage(documentId, 0);
      }
      
    } catch (err) {
      console.error('PDF extraction error:', err);
      setError('Failed to extract PDF info: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [loadOCRTextForPage]);

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

  // Load OCR text when page changes (but not when pdfInfo changes to prevent loops)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (pdfInfo && documentId && pdfInfo.pages && pdfInfo.pages[selectedPage]) {
      loadOCRTextForPage(documentId, selectedPage);
    }
  }, [selectedPage, documentId]); // Intentionally excluding pdfInfo and loadOCRTextForPage to prevent infinite loops

  const handlePageChange = (pageIndex) => {
    if (hasUnsavedChanges) {
      if (!window.confirm('You have unsaved changes. Are you sure you want to switch pages?')) {
        return;
      }
    }
    setSelectedPage(pageIndex);
  };

  const handleTextBlockClick = (textBlock) => {
    setActiveTextBlock(textBlock.id);
  };

  const updateTextBlock = (id, newText) => {
    setOcrTextBlocks(prev => prev.map(block => {
      if (block.id === id) {
        const isEdited = newText !== block.originalText;
        return { 
          ...block, 
          editedText: newText,
          isEdited
        };
      }
      return block;
    }));
    setHasUnsavedChanges(true);
  };

  const saveChanges = async () => {
    try {
      setLoading(true);
      
      // Get only the edited text blocks
      const editedBlocks = ocrTextBlocks.filter(block => block.isEdited);
      
      if (editedBlocks.length === 0) {
        setError('No changes to save');
        return;
      }
      
      console.log('Saving OCR corrections:', editedBlocks);
      
      // Here you would call an API to save the OCR corrections
      // For now, we'll just simulate success
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mark all blocks as saved
      setOcrTextBlocks(prev => prev.map(block => ({
        ...block,
        originalText: block.editedText,
        isEdited: false
      })));
      
      setHasUnsavedChanges(false);
      setError(null);
      
      console.log('OCR corrections saved successfully');
      
    } catch (err) {
      console.error('Save error:', err);
      setError('Failed to save changes: ' + err.message);
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
  const activeBlock = ocrTextBlocks.find(block => block.id === activeTextBlock);
  const editedBlocksCount = ocrTextBlocks.filter(block => block.isEdited).length;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">PDF OCR Editor</h1>
          <p className="text-gray-600">
            Edit the OCR'd text directly on the PDF to correct recognition errors.
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
                {hasUnsavedChanges && (
                  <span className="ml-2 text-orange-600 font-medium">
                    • {editedBlocksCount} unsaved changes
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Controls and Page Navigation */}
        {pdfInfo && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            {/* Zoom Controls at Top */}
            <div className="flex items-center justify-between mb-4 pb-4 border-b">
              <div className="flex items-center space-x-4">
                <h3 className="text-lg font-semibold">View Controls</h3>
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium text-gray-700">Zoom:</label>
                  <button
                    onClick={() => setZoomLevel(prev => Math.max(0.25, prev - 0.25))}
                    className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-sm"
                    disabled={zoomLevel <= 0.25}
                  >
                    −
                  </button>
                  <span className="text-sm font-mono min-w-[60px] text-center">
                    {Math.round(zoomLevel * 100)}%
                  </span>
                  <button
                    onClick={() => setZoomLevel(prev => Math.min(3, prev + 0.25))}
                    className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-sm"
                    disabled={zoomLevel >= 3}
                  >
                    +
                  </button>
                  <button
                    onClick={() => {
                      setZoomLevel(1);
                      setPanOffset({ x: 0, y: 0 });
                    }}
                    className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm"
                  >
                    Reset View
                  </button>
                </div>
              </div>
              {hasUnsavedChanges && (
                <button
                  onClick={saveChanges}
                  disabled={loading}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 text-sm font-medium"
                >
                  {loading ? 'Saving...' : `Save ${editedBlocksCount} Changes`}
                </button>
              )}
            </div>

            {/* Page Navigation */}
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">Pages</h3>
              <div className="text-sm text-gray-600">
                {ocrTextBlocks.length} text blocks on current page
              </div>
            </div>
            <div className="flex space-x-3 overflow-x-auto pb-2">
              {pdfInfo.pages.map((page, index) => (
                <div
                  key={index}
                  className={`flex-shrink-0 p-3 rounded-lg cursor-pointer transition-colors border-2 ${
                    selectedPage === index
                      ? 'bg-blue-100 border-blue-500'
                      : 'bg-gray-50 hover:bg-gray-100 border-gray-200'
                  }`}
                  onClick={() => handlePageChange(index)}
                >
                  <div className="font-medium text-sm text-center">Page {index + 1}</div>
                  <div className="text-xs text-gray-500 text-center mt-1">
                    {Math.round(page.page_width)} × {Math.round(page.page_height)}
                  </div>
                  {page.text_blocks && (
                    <div className="text-xs text-blue-600 text-center mt-1">
                      {page.text_blocks.length} text blocks
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main Canvas Area */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">
              {currentPage ? `Page ${selectedPage + 1} - OCR Text Editor` : 'Select a page'}
            </h2>
            <div className="text-sm text-gray-500">
              {currentPage && `${ocrTextBlocks.length} text blocks`}
            </div>
          </div>

          {currentPage ? (
            <div className="relative border rounded-lg overflow-hidden" style={{ width: '800px', height: '600px', flexShrink: 0 }}>
              <div 
                className="absolute inset-0 overflow-auto bg-gray-100"
                style={{ cursor: isPanning ? 'grabbing' : 'default' }}
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
                  
                  {/* OCR Text overlays */}
                  {ocrTextBlocks.map(block => {
                    const [x1, y1, x2, y2] = block.bbox;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    
                    return (
                      <div
                        key={block.id}
                        className={`absolute border-2 ${
                          activeTextBlock === block.id 
                            ? 'border-blue-600 bg-blue-100' 
                            : block.isEdited
                            ? 'border-orange-500 bg-orange-100'
                            : 'border-green-500 bg-green-100 hover:border-green-600 hover:bg-green-200'
                        } transition-colors shadow-sm`}
                        style={{
                          left: `${x1}px`,
                          top: `${y1}px`,
                          width: `${width}px`,
                          height: `${height}px`,
                          zIndex: 10,
                          cursor: 'text',
                          opacity: 1,
                          minHeight: '20px',
                          minWidth: '20px'
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleTextBlockClick(block);
                        }}
                      >
                        <textarea
                          value={block.editedText}
                          onChange={(e) => updateTextBlock(block.id, e.target.value)}
                          className="w-full h-full resize-none border-none outline-none p-1 text-xs font-medium"
                          style={{
                            fontSize: `${Math.max(10, Math.min(18, height * 0.7))}px`,
                            color: '#1f2937',
                            fontFamily: 'Arial, sans-serif',
                            lineHeight: '1.2',
                            backgroundColor: 'rgba(255, 255, 255, 0.9)',
                            borderRadius: '2px'
                          }}
                          placeholder="Click to edit OCR text..."
                          onFocus={() => setActiveTextBlock(block.id)}
                        />
                        
                        {/* Confidence indicator */}
                        <div 
                          className="absolute -top-2 -right-2 w-4 h-4 rounded-full text-xs flex items-center justify-center font-bold shadow-sm"
                          style={{
                            backgroundColor: block.confidence > 0.8 ? '#10b981' : block.confidence > 0.6 ? '#f59e0b' : '#ef4444',
                            color: 'white',
                            fontSize: '10px',
                            border: '1px solid white'
                          }}
                          title={`OCR Confidence: ${(block.confidence * 100).toFixed(0)}%`}
                        >
                          {block.confidence > 0.8 ? '✓' : block.confidence > 0.6 ? '?' : '!'}
                        </div>
                      </div>
                    );
                  })}
                  
                  {/* Page info overlay */}
                  <div className="absolute top-4 left-4 text-gray-600 text-sm bg-white bg-opacity-90 px-3 py-2 rounded shadow-sm border z-20">
                    <div className="font-medium">PDF Page {selectedPage + 1}</div>
                    <div className="text-xs text-gray-500">
                      {Math.round(currentPage.page_width)} × {Math.round(currentPage.page_height)} pts
                    </div>
                    <div className="text-xs text-blue-600">
                      {ocrTextBlocks.length} OCR blocks
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Instructions overlay */}
              <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white text-xs p-2 rounded max-w-xs">
                <div>• Click text blocks to edit OCR text</div>
                <div>• Green: Original OCR text</div>
                <div>• Orange: Modified text</div>
                <div>• Blue: Currently selected</div>
                <div>• Ctrl+drag to pan, use controls to zoom</div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg" style={{ height: '600px' }}>
              <div className="text-center">
                <p className="text-gray-500">Select a page to start editing OCR text</p>
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
              <h3 className="text-sm font-semibold text-gray-800">OCR Editor Controls</h3>
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

            {/* Active Text Block Info */}
            {activeBlock && (
              <div className="mb-3 p-2 bg-gray-50 rounded">
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Selected Text Block
                </label>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>Confidence: {(activeBlock.confidence * 100).toFixed(0)}%</div>
                  <div>Status: {activeBlock.isEdited ? 'Modified' : 'Original'}</div>
                  <div>Length: {activeBlock.editedText.length} chars</div>
                </div>
              </div>
            )}

            {/* Save Controls */}
            {hasUnsavedChanges && (
              <div className="mb-3">
                <button
                  onClick={saveChanges}
                  disabled={loading}
                  className="w-full px-2 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 text-xs font-medium"
                >
                  {loading ? 'Saving...' : `Save ${editedBlocksCount} Changes`}
                </button>
              </div>
            )}

            {/* Status */}
            <div className="text-xs text-gray-600 space-y-1 pt-3 border-t">
              <div>Text Blocks: {ocrTextBlocks.length}</div>
              <div>Modified: {editedBlocksCount}</div>
              <div>Active: {activeBlock ? 'Yes' : 'None'}</div>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span>Loading PDF and OCR data...</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PDFOCREditor;
