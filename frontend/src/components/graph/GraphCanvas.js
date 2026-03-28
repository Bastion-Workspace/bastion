/**
 * Shared Cytoscape canvas wrapper: mount/destroy lifecycle, fcose layout, stylesheet,
 * and data-mapped visual encoding. Used by FileRelationGraph, EntityRelationGraph, UnifiedKnowledgeGraph.
 */
import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import cytoscape from 'cytoscape';
import fcose from 'cytoscape-fcose';
import { Box } from '@mui/material';

cytoscape.use(fcose);

const DEFAULT_STYLE = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': 10,
      color: '#333',
      'text-max-width': 80,
      'text-wrap': 'ellipsis',
      'text-shadow': '0 1px 2px rgba(0,0,0,0.15)',
    },
  },
  {
    selector: 'node[type = "file"]',
    style: {
      width: 'mapData(labelLength, 5, 70, 100, 380)',
      height: 'mapData(labelLength, 5, 70, 28, 52)',
      'background-color': 'data(color)',
      'border-width': 1,
      'border-color': 'rgba(0,0,0,0.18)',
      'border-style': 'solid',
      shape: 'round-rectangle',
      'text-max-width': 200,
      'text-wrap': 'ellipsis',
      'underlay-color': 'rgba(0,0,0,0.14)',
      'underlay-padding': 4,
      'underlay-opacity': 1,
      'underlay-shape': 'round-rectangle',
    },
  },
  {
    selector: 'node[type = "entity"]',
    style: {
      width: 'mapData(labelLength, 3, 50, 44, 200)',
      height: 'mapData(labelLength, 3, 50, 28, 48)',
      'background-color': 'data(color)',
      'border-color': 'data(typeColor)',
      'border-width': 2,
      'border-style': 'solid',
      shape: 'ellipse',
      'text-max-width': 180,
      'text-wrap': 'ellipsis',
      'underlay-color': 'rgba(0,0,0,0.12)',
      'underlay-padding': 4,
      'underlay-opacity': 1,
      'underlay-shape': 'ellipse',
    },
  },
  {
    selector: 'node[type = "document"]',
    style: {
      width: 40,
      height: 28,
      'background-color': 'data(color)',
      'border-width': 1,
      'border-color': 'rgba(0,0,0,0.18)',
      shape: 'round-rectangle',
      'underlay-color': 'rgba(0,0,0,0.14)',
      'underlay-padding': 3,
      'underlay-opacity': 1,
      'underlay-shape': 'round-rectangle',
    },
  },
  {
    selector: 'node[node_type = "file"]',
    style: {
      width: 'mapData(labelLength, 5, 70, 100, 380)',
      height: 'mapData(labelLength, 5, 70, 28, 52)',
      'background-color': 'data(color)',
      'border-width': 1,
      'border-color': 'rgba(0,0,0,0.18)',
      'border-style': 'solid',
      shape: 'round-rectangle',
      'text-max-width': 200,
      'text-wrap': 'ellipsis',
      'underlay-color': 'rgba(0,0,0,0.14)',
      'underlay-padding': 4,
      'underlay-opacity': 1,
      'underlay-shape': 'round-rectangle',
    },
  },
  {
    selector: 'node[node_type = "entity"]',
    style: {
      width: 'mapData(labelLength, 3, 50, 44, 200)',
      height: 'mapData(labelLength, 3, 50, 28, 48)',
      'background-color': 'data(color)',
      'border-color': 'data(typeColor)',
      'border-width': 2,
      'border-style': 'solid',
      shape: 'ellipse',
      'text-max-width': 180,
      'text-wrap': 'ellipsis',
      'underlay-color': 'rgba(0,0,0,0.12)',
      'underlay-padding': 4,
      'underlay-opacity': 1,
      'underlay-shape': 'ellipse',
    },
  },
  { selector: 'edge', style: { 'curve-style': 'bezier', 'target-arrow-color': '#999', 'target-arrow-shape': 'triangle', opacity: 0.85 } },
  { selector: 'edge[edge_type = "co_occurs"]', style: { width: 'mapData(weight, 1, 10, 1, 5)' } },
  { selector: 'edge[edge_type = "file_link"]', style: { width: 'mapData(weight, 1, 10, 1.5, 4)' } },
  { selector: 'edge[edge_type = "mentions"]', style: { width: 1.5 } },
  { selector: '.dimmed', style: { opacity: 0.15 } },
  { selector: 'node.hovered', style: { 'border-width': 3, 'border-color': '#1976d2', 'border-style': 'solid', opacity: 1 } },
  { selector: 'edge.hovered', style: { opacity: 1, width: 3 } },
];

const FCOSE_OPTIONS = {
  name: 'fcose',
  animate: true,
  animationDuration: 600,
  quality: 'proof',
  nodeRepulsion: 6000,
  idealEdgeLength: 80,
  edgeElasticity: 0.45,
  fit: true,
  padding: 40,
};

export const FCOSE_OPTIONS_LOOSE = {
  ...FCOSE_OPTIONS,
  nodeRepulsion: 9000,
  idealEdgeLength: 120,
};

/**
 * Build Cytoscape elements from nodes and edges (API shape).
 * Nodes: { id, label, type?, node_type?, degree?, doc_count?, color?, typeColor? }
 * labelLength is added so file nodes can be sized by name length.
 * Edges: { source, target, weight?, edge_type? }
 */
function buildElements(nodes, edges) {
  const cyNodes = (nodes || []).map((n) => {
    const label = n.label ?? n.title ?? n.id ?? '';
    const labelLength = typeof label === 'string' ? label.length : 0;
    return {
      group: 'nodes',
      data: {
        id: n.id,
        label,
        labelLength: Math.max(1, Math.min(labelLength, 70)),
        ...n,
      },
    };
  });
  const seen = new Set();
  const cyEdges = (edges || []).map((e) => {
    const id = `e-${e.source}-${e.target}-${seen.size}`;
    seen.add(id);
    return {
      group: 'edges',
      data: {
        id,
        source: e.source,
        target: e.target,
        weight: e.weight ?? 1,
        edge_type: e.edge_type ?? 'link',
        ...e,
      },
    };
  });
  return [...cyNodes, ...cyEdges];
}

export default function GraphCanvas({
  nodes = [],
  edges = [],
  style = DEFAULT_STYLE,
  layoutOptions = {},
  onNodeClick,
  onNodeDoubleClick,
  onPaneClick,
  selectedNodeId,
  focusedNodeIds,
  theme,
  initialViewport,
  onViewportChange,
  className,
  sx,
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const lastLayoutSigRef = useRef(null);

  const elements = useMemo(() => buildElements(nodes, edges), [nodes, edges]);

  const elementsLayoutSig = useMemo(() => {
    if (!elements.length) return '';
    const nodeIds = [...new Set(elements.filter((el) => el.group === 'nodes').map((el) => el.data.id))].sort().join(',');
    const edgeCount = elements.filter((el) => el.group === 'edges').length;
    return `${nodeIds}|${edgeCount}`;
  }, [elements]);

  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      style: style || DEFAULT_STYLE,
      minZoom: 0.1,
      maxZoom: 4,
      wheelSensitivity: 1.2,
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const sig = elementsLayoutSig;
    const structureChanged = sig !== lastLayoutSigRef.current;
    lastLayoutSigRef.current = sig;

    if (elements.length === 0) {
      cy.elements().remove();
      return;
    }

    if (structureChanged) {
      cy.elements().remove();
      cy.add(elements);
      cy.layout({ ...FCOSE_OPTIONS, ...layoutOptions }).run();
    } else {
      elements.forEach((el) => {
        if (el.group !== 'nodes') return;
        const node = cy.getElementById(el.data.id);
        if (node.length) node.data(el.data);
      });
    }
  }, [elements, elementsLayoutSig, layoutOptions]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    if (focusedNodeIds && focusedNodeIds.size > 0) {
      cy.nodes().addClass('dimmed');
      cy.nodes().filter((n) => focusedNodeIds.has(n.id())).removeClass('dimmed');
      cy.edges().addClass('dimmed');
      cy.edges().filter((e) => focusedNodeIds.has(e.source().id()) && focusedNodeIds.has(e.target().id())).removeClass('dimmed');
    } else {
      cy.elements().removeClass('dimmed');
    }
  }, [focusedNodeIds]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !initialViewport) return;
    const { x = 0, y = 0, zoom } = initialViewport;
    if (typeof zoom === 'number') {
      cy.zoom(zoom);
      cy.pan({ x, y });
    }
  }, [initialViewport]);

  const handleTap = useCallback(
    (evt) => {
      if (evt.target === cyRef.current) {
        onPaneClick?.();
        return;
      }
      if (evt.target.isNode() && onNodeClick) {
        if (evt.stopPropagation) evt.stopPropagation();
        onNodeClick(evt.target.data());
      }
    },
    [onNodeClick, onPaneClick]
  );

  const handleDblTap = useCallback(
    (evt) => {
      if (evt.target.isNode() && onNodeDoubleClick) {
        onNodeDoubleClick(evt.target.data());
      }
    },
    [onNodeDoubleClick]
  );

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.on('tap', handleTap);
    cy.on('dbltap', 'node', handleDblTap);
    const onMouseover = (evt) => {
      const node = evt.target;
      node.addClass('hovered');
      node.connectedEdges().addClass('hovered');
    };
    const onMouseout = (evt) => {
      const node = evt.target;
      node.removeClass('hovered');
      node.connectedEdges().removeClass('hovered');
    };
    cy.on('mouseover', 'node', onMouseover);
    cy.on('mouseout', 'node', onMouseout);
    return () => {
      cy.off('tap', handleTap);
      cy.off('dbltap', 'node', handleDblTap);
      cy.off('mouseover', 'node', onMouseover);
      cy.off('mouseout', 'node', onMouseout);
    };
  }, [handleTap, handleDblTap]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !onViewportChange) return;
    const handler = () => {
      const pan = cy.pan();
      const zoom = cy.zoom();
      onViewportChange({ x: pan.x, y: pan.y, zoom });
    };
    cy.on('pan zoom', handler);
    return () => cy.off('pan zoom', handler);
  }, [onViewportChange]);

  return (
    <Box
      ref={containerRef}
      className={className}
      sx={{
        width: '100%',
        height: '100%',
        minHeight: 400,
        bgcolor: theme?.palette?.background?.default ?? '#fafafa',
        ...sx,
      }}
    />
  );
}
