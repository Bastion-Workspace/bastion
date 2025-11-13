import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';

// PDF.js imports
import * as pdfjsLib from 'pdfjs-dist';
import 'pdfjs-dist/build/pdf.worker.entry';

// Set PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const PDFTextLayerEditor = () => {
  const { documentId } = useParams();
  const { user } = useAuth();
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [selectedPage, setSelectedPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // PDF rendering
  const pdfCanvasRef = useRef(null);
  const renderTaskRef = useRef(null);
  const viewportRef = useRef(null);
  
  // Map-style viewport state
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  
  // Text editing state
  const [textBlocks, setTextBlocks] = useState([]);
  const [selectedBlockId, setSelectedBlockId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [originalText, setOriginalText] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [savedBlocks, setSavedBlocks] = useState(new Set());
  
  // Debug mode
  const [debugMode, setDebugMode] = useState(false);
  
  // Textarea ref for auto-resize
  const textareaRef = useRef(null);
  
  // PDF.js state
  const [pdfDocument, setPdfDocument] = useState(null);
  const [pageCount, setPageCount] = useState(0);

  const loadPDFDocument = useCallback(async (documentId) => {
    try {
      console.log('Loading PDF document:', documentId);
      
      const pdfUrl = `/api/documents/${documentId}/pdf`;
      const loadingTask = pdfjsLib.getDocument(pdfUrl);
      const pdf = await loadingTask.promise;
      
      setPdfDocument(pdf);
      setPageCount(pdf.numPages);
      
      console.log('PDF loaded successfully:', pdf.numPages, 'pages');
      return pdf;
      
    } catch (err) {
      console.error('PDF loading error:', err);
      setError('Failed to load PDF: ' + err.message);
      return null;
    }
  }, []);

  const renderPDFPage = useCallback(async (pageNumber) => {
    if (!pdfDocument) return;
    
    try {
      // Cancel previous render if it exists
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }
      
      console.log('Rendering PDF page:', pageNumber);
      
      const page = await pdfDocument.getPage(pageNumber);
      
      // Use a base scale for the initial render, zoom will be handled by CSS transform
      const baseScale = 1.5;
      const viewport = page.getViewport({ scale: baseScale });
      
      const canvas = pdfCanvasRef.current;
      if (!canvas) return;
      
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      // Clear canvas
      context.clearRect(0, 0, canvas.width, canvas.height);
      
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      
      // Start new render and store the task
      renderTaskRef.current = page.render(renderContext);
      await renderTaskRef.current.promise;
      
      console.log('PDF page rendered successfully');
      
    } catch (err) {
      if (err.name !== 'RenderingCancelledException') {
        console.error('PDF rendering error:', err);
        setError('Failed to render PDF page: ' + err.message);
      }
    }
  }, [pdfDocument]);

  // Auto-resize textarea
  const autoResizeTextarea = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 400) + 'px';
    }
  }, []);

  useEffect(() => {
    autoResizeTextarea();
  }, [editingText, autoResizeTextarea]);

  // Define loadTextBlocks before useEffects
  const loadTextBlocks = useCallback(async (docId, pageNumber) => {
    try {
      console.log('Loading text blocks for page:', pageNumber);
      const response = await apiService.extractPdfTextLayer({
        document_id: docId,
        page_number: pageNumber
      });
      console.log('Text blocks response:', response);
      
      if (response && Array.isArray(response.text_blocks)) {
        setTextBlocks(response.text_blocks);
        console.log('Text blocks loaded:', response.text_blocks.length);
        console.log('First few blocks:', response.text_blocks.slice(0, 3));
      } else {
        console.log('No text blocks found or invalid response:', response);
        setTextBlocks([]);
      }
    } catch (err) {
      console.error('Error loading text blocks:', err);
      setTextBlocks([]);
    }
  }, []);

  // Load document data
  useEffect(() => {
    const loadDocument = async () => {
      if (!documentId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        // Load document metadata via document list
        const documentsResponse = await apiService.getDocuments(0, 1000);
        const document = documentsResponse.documents.find(doc => doc.document_id === documentId);
        
        if (!document) {
          throw new Error('Document not found');
        }
        
        console.log('Document loaded:', document);
        setSelectedDocument(document);
        
        // Load PDF
        const pdf = await loadPDFDocument(documentId);
        if (!pdf) return;
        
        // Load text blocks for the first page
        await loadTextBlocks(documentId, 1);
        
      } catch (err) {
        console.error('Error loading document:', err);
        setError('Failed to load document: ' + err.message);
      } finally {
        setLoading(false);
      }
    };
    
    loadDocument();
  }, [documentId, loadPDFDocument, loadTextBlocks]);

  // Render PDF when page changes
  useEffect(() => {
    if (pdfDocument && selectedPage) {
      renderPDFPage(selectedPage);
      loadTextBlocks(documentId, selectedPage);
    }
  }, [pdfDocument, selectedPage, renderPDFPage, documentId, loadTextBlocks]);

  const handleBlockClick = (block) => {
    console.log('🎯 Block clicked!', block);
    console.log('Block ID:', block.id);
    console.log('Block text:', block.text);
    console.log('Block bbox:', block.bbox);
    
    setSelectedBlockId(block.id);
    setEditingText(block.text || '');
    setOriginalText(block.text || '');
    setHasUnsavedChanges(false);
    
    console.log('✅ Block selection updated');
  };

  const handleTextChange = (e) => {
    setEditingText(e.target.value);
    setHasUnsavedChanges(e.target.value !== originalText);
  };

  const saveTextBlock = async () => {
    if (!selectedBlockId || !selectedDocument) return;
    
    try {
      setLoading(true);
      
      await apiService.updatePdfTextBlock({
        document_id: selectedDocument.document_id,
        page_number: selectedPage,
        block_id: selectedBlockId,
        new_text: editingText
      });
      
      // Update the text block in our state
      setTextBlocks(prevBlocks =>
        prevBlocks.map(block =>
          block.id === selectedBlockId
            ? { ...block, text: editingText }
            : block
        )
      );
      
      // Mark as saved
      setSavedBlocks(prev => new Set([...prev, selectedBlockId]));
      setHasUnsavedChanges(false);
      setOriginalText(editingText);
      
      console.log('Text block saved successfully');
      
    } catch (err) {
      console.error('Error saving text block:', err);
      setError('Failed to save text block: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const reprocessDocument = async () => {
    if (!selectedDocument) return;
    
    try {
      setLoading(true);
      setError(null);
      
      console.log('Reprocessing document...');
      
      // Choose reprocessing endpoint based on user role
      // PDF text layer editing is typically for global documents (admin functionality)
      // but we'll use the appropriate endpoint based on user role for safety
      const isAdmin = user?.role === 'admin';
      const reprocessPromise = isAdmin
        ? apiService.reprocessDocument(selectedDocument.document_id)
        : apiService.reprocessUserDocument(selectedDocument.document_id);
      
      await reprocessPromise;
      
      // Reload text blocks after reprocessing
      await loadTextBlocks(selectedDocument.document_id, selectedPage);
      
      // Clear saved blocks since they might be different now
      setSavedBlocks(new Set());
      setSelectedBlockId(null);
      setEditingText('');
      setOriginalText('');
      setHasUnsavedChanges(false);
      
      console.log('Document reprocessed successfully');
      
    } catch (err) {
      console.error('Error reprocessing document:', err);
      setError('Failed to reprocess document: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Map-style viewport controls
  const handleViewportMouseDown = (e) => {
    if (e.target === viewportRef.current || e.target === pdfCanvasRef.current) {
      setIsPanning(true);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
  };

  const handleViewportMouseMove = (e) => {
    if (isPanning) {
      const deltaX = e.clientX - lastPanPoint.x;
      const deltaY = e.clientY - lastPanPoint.y;
      
      setPanOffset(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      
      setLastPanPoint({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
  };

  const handleViewportMouseUp = () => {
    setIsPanning(false);
  };

  const handleViewportWheel = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    const newZoom = Math.min(Math.max(0.3, zoomLevel + delta), 5.0);
    setZoomLevel(newZoom);
  }, [zoomLevel]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (viewport) {
      viewport.addEventListener('wheel', handleViewportWheel, { passive: false });
      return () => {
        viewport.removeEventListener('wheel', handleViewportWheel, { passive: false });
      };
    }
  }, [handleViewportWheel]);

  const resetViewport = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const goToPage = (pageNumber) => {
    if (pageNumber >= 1 && pageNumber <= pageCount) {
      setSelectedPage(pageNumber);
      // Reset viewport when changing pages
      resetViewport();
    }
  };

  if (!selectedDocument) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading document...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <div className="flex-1 grid grid-cols-[auto,1fr] overflow-hidden">
        {/* Left Panel - Text Editor */}
        <div className="w-96 bg-white border-r border-gray-300 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-gray-200 flex-shrink-0">
            <h2 className="text-lg font-semibold text-gray-800">Text Block Editor</h2>
            <p className="text-sm text-gray-600 mt-1">{selectedDocument.filename}</p>
            
            {/* Page Navigation */}
            <div className="flex items-center space-x-2 mt-3">
              <button
                onClick={() => goToPage(selectedPage - 1)}
                disabled={selectedPage <= 1}
                className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300 text-sm"
              >
                ← Prev
              </button>
              <span className="text-sm text-gray-600">
                Page {selectedPage} of {pageCount}
              </span>
              <button
                onClick={() => goToPage(selectedPage + 1)}
                disabled={selectedPage >= pageCount}
                className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300 text-sm"
              >
                Next →
              </button>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-2 mt-3">
              <button
                onClick={resetViewport}
                className="px-3 py-1 bg-gray-500 text-white rounded text-sm"
              >
                🎯 Reset View
              </button>
              <button
                onClick={() => setDebugMode(!debugMode)}
                className={`px-3 py-1 rounded text-sm ${
                  debugMode 
                    ? 'bg-red-500 text-white' 
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                Debug {debugMode ? 'ON' : 'OFF'}
              </button>
            </div>

            <div className="text-xs text-gray-500 mt-2">
              Zoom: {Math.round(zoomLevel * 100)}% | Pan: {Math.round(panOffset.x)}, {Math.round(panOffset.y)}
            </div>
          </div>

          {/* Text Editor Area */}
          <div className="flex-1 p-4 overflow-auto">
            {selectedBlockId ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Edit Text Block
                  </label>
                  <textarea
                    ref={textareaRef}
                    value={editingText}
                    onChange={handleTextChange}
                    className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Edit the text content..."
                    style={{ minHeight: '120px' }}
                  />
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={saveTextBlock}
                    disabled={!hasUnsavedChanges || loading}
                    className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300 hover:bg-blue-600"
                  >
                    {savedBlocks.has(selectedBlockId) && !hasUnsavedChanges ? '✓ Saved' : 'Save Changes'}
                  </button>
                  <button
                    onClick={() => {
                      setSelectedBlockId(null);
                      setEditingText('');
                      setOriginalText('');
                      setHasUnsavedChanges(false);
                    }}
                    className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
                  >
                    Clear
                  </button>
                </div>

                {originalText && (
                  <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Original Text:
                    </label>
                    <div className="text-sm text-gray-600 whitespace-pre-wrap break-words">
                      {originalText}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500 mt-8">
                <p>Click on a highlighted text block in the PDF to edit it</p>
                <p className="text-sm mt-2">Text blocks: {textBlocks.length}</p>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="p-4 border-t border-gray-200 flex-shrink-0">
            <button
              onClick={reprocessDocument}
              disabled={loading}
              className="w-full px-4 py-3 bg-green-500 text-white rounded-lg disabled:bg-gray-300 hover:bg-green-600"
            >
              🔄 Reprocess Document
            </button>
          </div>

          {error && (
            <div className="mt-2 p-2 bg-red-100 border border-red-400 text-red-700 rounded flex-shrink-0">
              {error}
            </div>
          )}
        </div>

        {/* Right Panel - PDF Viewport (Map-style) */}
        <div className="flex-1 flex flex-col items-center justify-center bg-gray-50 p-4">
          {/* Viewport Container - Fixed Size and Position */}
          <div 
            ref={viewportRef}
            className="relative border-2 border-gray-400 overflow-hidden bg-white shadow-lg"
            style={{ 
              width: '800px',
              height: '600px',
              cursor: isPanning ? 'grabbing' : 'grab',
              position: 'relative',
              flexShrink: 0
            }}
            onMouseDown={handleViewportMouseDown}
            onMouseMove={handleViewportMouseMove}
            onMouseUp={handleViewportMouseUp}
            onMouseLeave={handleViewportMouseUp}
          >
            {/* PDF Content with Transform */}
            <div
              style={{
                transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
                transformOrigin: '0 0',
                transition: isPanning ? 'none' : 'transform 0.1s ease-out',
                position: 'absolute',
                top: 0,
                left: 0
              }}
            >
              {/* PDF Canvas */}
              <canvas
                ref={pdfCanvasRef}
                style={{
                  display: 'block',
                  border: '1px solid #ccc'
                }}
              />

                             {/* Text Block Overlays */}
               {textBlocks.map((block) => {
                 const x1 = block.bbox?.[0] || 0;
                 const y1 = block.bbox?.[1] || 0;
                 const x2 = block.bbox?.[2] || 0;
                 const y2 = block.bbox?.[3] || 0;
                 const width = x2 - x1;
                 const height = y2 - y1;

                 return (
                   <div
                     key={block.id}
                     className={`absolute border-2 cursor-pointer hover:opacity-90 ${
                       selectedBlockId === block.id
                         ? 'border-blue-500 bg-blue-200' 
                         : savedBlocks.has(block.id)
                         ? 'border-green-500 bg-green-200'
                         : debugMode
                         ? 'border-red-500 bg-red-200'
                         : 'border-gray-400 bg-gray-200'
                     }`}
                     style={{
                       left: `${x1}px`,
                       top: `${y1}px`,
                       width: `${width}px`,
                       height: `${height}px`,
                       opacity: debugMode ? 0.7 : 0.3,
                       zIndex: 10,
                       minWidth: '2px',
                       minHeight: '2px'
                     }}
                     onClick={(e) => {
                       e.preventDefault();
                       e.stopPropagation();
                       console.log('Text block clicked!', block.id, isPanning);
                       handleBlockClick(block);
                     }}
                     onMouseDown={(e) => {
                       e.preventDefault();
                       e.stopPropagation();
                       setIsPanning(false); // Prevent panning when clicking text
                     }}
                     onMouseUp={(e) => {
                       e.preventDefault();
                       e.stopPropagation();
                     }}
                     title={`Click to edit: ${block.text?.substring(0, 50) || 'Loading...'}...`}
                   >
                     {debugMode && (
                       <div className="absolute -top-5 left-0 text-xs text-red-600 bg-white px-1 rounded">
                         {block.id}
                       </div>
                     )}
                   </div>
                 );
               })}
            </div>
          </div>

          {/* Debug Info Panel */}
          <div className="px-4 pb-4 space-y-2">
            <div className="text-sm text-gray-600">
              Mouse wheel to zoom • Click and drag to pan • Click text blocks to edit
            </div>
            <div className="text-xs text-gray-500 bg-gray-100 p-2 rounded">
              <div>Text Blocks: {textBlocks.length} loaded</div>
              <div>Zoom: {Math.round(zoomLevel * 100)}%</div>
              <div>Pan: ({Math.round(panOffset.x)}, {Math.round(panOffset.y)})</div>
              <div>Is Panning: {isPanning ? 'Yes' : 'No'}</div>
              <div>Selected Block: {selectedBlockId || 'None'}</div>
            </div>
          </div>
        </div>
      </div>

      {loading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-4 rounded-lg shadow-lg">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>Loading...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PDFTextLayerEditor; 