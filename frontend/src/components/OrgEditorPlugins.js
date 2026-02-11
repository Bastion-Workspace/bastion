import { Decoration, ViewPlugin, EditorView } from '@codemirror/view';
import { foldService, foldState, foldEffect, unfoldEffect, codeFolding } from '@codemirror/language';

const TODO_STATES = ['TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD'];
const DONE_STATES = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];

export function getOrgHeaderLevel(lineText) {
  const match = lineText.match(/^(\*+)\s/);
  return match ? match[1].length : 0;
}

export function findOrgFoldRange(state, startLine) {
  const doc = state.doc;
  
  if (startLine < 1 || startLine > doc.lines) {
    return null;
  }
  
  try {
    const startLineObj = doc.line(startLine);
    const startLineText = startLineObj.text;
    const startLevel = getOrgHeaderLevel(startLineText);
    
    if (startLevel === 0) return null;
    
    let endLine = startLine;
    for (let i = startLine + 1; i <= doc.lines; i++) {
      const lineObj = doc.line(i);
      const lineText = lineObj.text;
      const level = getOrgHeaderLevel(lineText);
      if (level > 0 && level <= startLevel) {
        endLine = i - 1;
        break;
      }
      endLine = i;
    }
    
    if (endLine === startLine) return null;
    
    const from = startLineObj.to;
    const endLineObj = doc.line(endLine);
    const to = endLineObj.to;
    
    // Strict validation for folds (replacement decorations must be non-empty and valid)
    if (typeof from !== 'number' || typeof to !== 'number' || isNaN(from) || isNaN(to)) {
       console.warn('❌ ROOSEVELT: Invalid fold positions (NaN/type)', { from, to });
       return null;
    }
    if (from < 0 || to > doc.length) {
       console.warn('❌ ROOSEVELT: Fold range out of bounds', { from, to, docLen: doc.length });
       return null;
    }
    if (from >= to) {
       // Folds must replace something. from == to is empty replacement (widget), but fold implies hiding content.
       // Usually we hide at least one char or newline?
       // from = end of header line. to = end of end line.
       // If endLine > startLine, range includes at least one newline.
       // So from < to should be true.
       console.warn('❌ ROOSEVELT: Empty or inverted fold range', { from, to });
       return null;
    }
    
    return { from, to };
  } catch (err) {
    console.warn(`Error finding fold range for line ${startLine}:`, err);
    return null;
  }
}

// Export codeFolding for use in editor extensions
export { codeFolding };

export const orgFoldService = foldService.of((state, lineStart, lineEnd) => {
  const ranges = [];
  const doc = state.doc;
  const start = Math.max(1, lineStart);
  const end = Math.min(doc.lines, lineEnd);
  
  for (let line = start; line <= end; line++) {
    try {
      const lineObj = doc.line(line);
      if (getOrgHeaderLevel(lineObj.text) > 0) {
        const foldRange = findOrgFoldRange(state, line);
        if (foldRange) {
          ranges.push(foldRange);
        }
      }
    } catch (err) {
      continue;
    }
  }
  return ranges;
});

export const orgDecorationsPlugin = ViewPlugin.fromClass(class {
  constructor(view) {
    this.decorations = Decoration.none;
    this.updateDecos(view);
  }
  update(update) {
    let prevSize = 0;
    let curSize = 0;
    try {
      // foldState returns a RangeSet directly, not an object with .ranges
      const prev = update.startState.field(foldState, false);
      if (prev) prevSize = prev.size || 0;
      
      const cur = update.state.field(foldState, false);
      if (cur) curSize = cur.size || 0;
    } catch(e) {}
    
    const foldStateChanged = prevSize !== curSize;
    
    if (update.docChanged || update.viewportChanged || foldStateChanged) {
      this.updateDecos(update.view);
    }
  }
  updateDecos(view) {
    const { doc } = view.state;
    const decos = [];
    
    const validate = (from, to) => {
      if (typeof from !== 'number' || typeof to !== 'number' || isNaN(from) || isNaN(to)) return false;
      if (from < 0 || to < 0) return false;
      if (from > to) return false; // Marks can be empty (from==to)
      if (from > doc.length || to > doc.length) return false;
      return true;
    };
    
    const addMark = (mark, from, to) => {
      if (!validate(from, to)) return;
      try { decos.push(mark.range(from, to)); } catch(e) {}
    };
    
    const addLine = (lineDeco, pos) => {
      if (typeof pos !== 'number' || isNaN(pos) || pos < 0 || pos > doc.length) return;
      try { decos.push(lineDeco.range(pos)); } catch(e) {}
    };

    for (let i = 1; i <= doc.lines; i++) {
      try {
        const line = doc.line(i);
        const start = line.from;
        const text = line.text;
        
        const head = text.match(/^(\*+)\s+(.*)$/);
        if (head) {
          const level = head[1].length;
          addLine(Decoration.line({ class: `org-heading org-level-${level}` }), start);
          
          const rest = head[2] || '';
          const first = (rest.split(/\s+/)[0] || '').toUpperCase();
          if (first) {
            const tStart = start + head[1].length + 1;
            const tEnd = tStart + first.length;
            if (TODO_STATES.includes(first)) {
              addMark(Decoration.mark({ class: 'org-todo-mark' }), tStart, tEnd);
            } else if (DONE_STATES.includes(first)) {
              addMark(Decoration.mark({ class: 'org-done-mark' }), tStart, tEnd);
            }
          }
        }
        
        const cbMatch = text.match(/^(\s*[-+*]\s+)\[( |x|X|-)\]/);
        if (cbMatch) {
          const cbStart = start + cbMatch[1].length;
          const cbEnd = start + cbMatch[0].length;
          addMark(Decoration.mark({ class: 'org-checkbox' }), cbStart, cbEnd);
        }
        
        const linkRegex = /\[\[([^\]]+)\](?:\[([^\]]+)\])?\]/g;
        let match;
        while ((match = linkRegex.exec(text)) !== null) {
          const lStart = start + match.index;
          const lEnd = lStart + match[0].length;
          addMark(Decoration.mark({ class: 'org-link' }), lStart, lEnd);
        }
        
        // Decorate progress indicators [n/m] or [n%]
        const progressFractionRegex = /\[(\d+)\/(\d+)\]/g;
        while ((match = progressFractionRegex.exec(text)) !== null) {
          const pStart = start + match.index;
          const pEnd = pStart + match[0].length;
          addMark(Decoration.mark({ class: 'org-progress-indicator' }), pStart, pEnd);
        }
        
        const progressPercentRegex = /\[(\d+)%\]/g;
        while ((match = progressPercentRegex.exec(text)) !== null) {
          const pStart = start + match.index;
          const pEnd = pStart + match[0].length;
          addMark(Decoration.mark({ class: 'org-progress-indicator' }), pStart, pEnd);
        }
      } catch (e) { continue; }
    }
    
    try {
      this.decorations = Decoration.set(decos, true);
    } catch (e) {
      console.error('Failed to set decorations:', e);
      this.decorations = Decoration.none;
    }
  }
}, { decorations: v => v.decorations });

// Helper functions for checkbox tree traversal
function getIndentLevel(lineText) {
  const match = lineText.match(/^(\s*)/);
  return match ? match[1].length : 0;
}

function getCheckboxState(lineText) {
  const match = lineText.match(/\[( |x|X|-)\]/);
  if (!match) return null;
  const state = match[1].toUpperCase();
  if (state === 'X') return 'X';
  if (state === '-') return '-';
  return ' ';
}

function setCheckboxState(lineText, newState) {
  return lineText.replace(/\[( |x|X|-)\]/i, `[${newState}]`);
}

function hasCheckbox(lineText) {
  return /^(\s*[-+*]\s+)\[( |x|X|-)\]/i.test(lineText);
}

function findParentCheckbox(doc, lineNum) {
  if (lineNum < 1) return null;
  
  const currentLine = doc.line(lineNum);
  const currentIndent = getIndentLevel(currentLine.text);
  
  // Search backwards for checkbox with lower indent
  for (let i = lineNum - 1; i >= 1; i--) {
    const line = doc.line(i);
    const lineText = line.text;
    
    // Stop if we hit a heading (checkboxes belong to headings, not other checkboxes)
    if (getOrgHeaderLevel(lineText) > 0) {
      break;
    }
    
    if (hasCheckbox(lineText)) {
      const indent = getIndentLevel(lineText);
      if (indent < currentIndent) {
        return i;
      }
    }
  }
  
  return null;
}

function findChildCheckboxes(doc, parentLineNum) {
  if (parentLineNum < 1 || parentLineNum > doc.lines) return [];
  
  const parentLine = doc.line(parentLineNum);
  const parentIndent = getIndentLevel(parentLine.text);
  const children = [];
  
  // Search forwards for direct children (same indent level + 1)
  for (let i = parentLineNum + 1; i <= doc.lines; i++) {
    const line = doc.line(i);
    const lineText = line.text;
    
    // Stop if we hit a heading at same or lower level
    const headingLevel = getOrgHeaderLevel(lineText);
    if (headingLevel > 0) {
      break;
    }
    
    if (hasCheckbox(lineText)) {
      const indent = getIndentLevel(lineText);
      if (indent === parentIndent + 2) { // Direct child (2 spaces more)
        children.push(i);
      } else if (indent <= parentIndent) {
        // Sibling or parent - stop searching
        break;
      }
    }
  }
  
  return children;
}

function findParentHeading(doc, lineNum) {
  // Search backwards for heading
  for (let i = lineNum; i >= 1; i--) {
    const line = doc.line(i);
    const level = getOrgHeaderLevel(line.text);
    if (level > 0) {
      return i;
    }
  }
  return null;
}

function parseProgressIndicator(headingText) {
  const fractionMatch = headingText.match(/\[(\d+)\/(\d+)\]/);
  if (fractionMatch) {
    return {
      type: 'fraction',
      position: fractionMatch.index,
      current: parseInt(fractionMatch[1]),
      total: parseInt(fractionMatch[2])
    };
  }
  
  const percentMatch = headingText.match(/\[(\d+)%\]/);
  if (percentMatch) {
    return {
      type: 'percent',
      position: percentMatch.index,
      percent: parseInt(percentMatch[1])
    };
  }
  
  return null;
}

function calculateProgress(doc, headingLineNum) {
  const children = [];
  
  // Find all checkboxes under this heading
  for (let i = headingLineNum + 1; i <= doc.lines; i++) {
    const line = doc.line(i);
    const lineText = line.text;
    
    // Stop if we hit another heading at same or higher level
    const headingLevel = getOrgHeaderLevel(lineText);
    if (headingLevel > 0) {
      const currentHeadingLevel = getOrgHeaderLevel(doc.line(headingLineNum).text);
      if (headingLevel <= currentHeadingLevel) {
        break;
      }
    }
    
    if (hasCheckbox(lineText)) {
      const state = getCheckboxState(lineText);
      children.push({ lineNum: i, state });
    }
  }
  
  const total = children.length;
  const checked = children.filter(c => c.state === 'X').length;
  
  return { checked, total, children };
}

function updateProgressIndicator(view, headingLineNum) {
  const doc = view.state.doc;
  const headingLine = doc.line(headingLineNum);
  const headingText = headingLine.text;
  
  const indicator = parseProgressIndicator(headingText);
  if (!indicator) return; // No indicator to update
  
  const progress = calculateProgress(doc, headingLineNum);
  
  if (progress.total === 0) return; // No checkboxes to count
  
  let newText = headingText;
  
  if (indicator.type === 'fraction') {
    newText = headingText.replace(/\[\d+\/\d+\]/, `[${progress.checked}/${progress.total}]`);
  } else if (indicator.type === 'percent') {
    const percent = Math.round((progress.checked / progress.total) * 100);
    newText = headingText.replace(/\[\d+%\]/, `[${percent}%]`);
  }
  
  if (newText !== headingText) {
    view.dispatch({
      changes: { from: headingLine.from, to: headingLine.to, insert: newText }
    });
  }
}

function updateParentCheckboxStates(view, startLineNum) {
  const doc = view.state.doc;
  const changes = [];
  
  // Update parent checkboxes recursively
  let currentLine = startLineNum;
  while (true) {
    const parentLineNum = findParentCheckbox(doc, currentLine);
    if (!parentLineNum) break;
    
    const parentLine = doc.line(parentLineNum);
    const children = findChildCheckboxes(doc, parentLineNum);
    
    if (children.length === 0) break;
    
    // Count states
    const states = children.map(childLineNum => {
      const childText = doc.line(childLineNum).text;
      return getCheckboxState(childText);
    });
    
    const allChecked = states.every(s => s === 'X');
    const allUnchecked = states.every(s => s === ' ');
    const newState = allChecked ? 'X' : (allUnchecked ? ' ' : '-');
    
    const currentState = getCheckboxState(parentLine.text);
    if (currentState !== newState) {
      const newText = setCheckboxState(parentLine.text, newState);
      changes.push({
        from: parentLine.from,
        to: parentLine.to,
        insert: newText
      });
    }
    
    // Also update progress indicators for parent heading
    const parentHeading = findParentHeading(doc, parentLineNum);
    if (parentHeading) {
      // Will update after all changes are applied
      setTimeout(() => updateProgressIndicator(view, parentHeading), 0);
    }
    
    currentLine = parentLineNum;
  }
  
  // Apply all changes in one transaction
  if (changes.length > 0) {
    view.dispatch({ changes });
  }
}

function toggleCheckboxAtCursor(view) {
  const state = view.state;
  const pos = state.selection.main.head;
  const line = state.doc.lineAt(pos);
  const lineText = line.text;
  
  // Check if current line has checkbox
  if (!hasCheckbox(lineText)) {
    // Search backwards for nearest checkbox
    for (let i = line.number - 1; i >= 1; i--) {
      const searchLine = state.doc.line(i);
      if (hasCheckbox(searchLine.text)) {
        const currentState = getCheckboxState(searchLine.text);
        const newState = currentState === 'X' ? ' ' : 'X';
        const newText = setCheckboxState(searchLine.text, newState);
        
        view.dispatch({
          changes: { from: searchLine.from, to: searchLine.to, insert: newText }
        });
        
        // Update parent checkboxes
        updateParentCheckboxStates(view, i);
        
        // Update progress indicators
        const parentHeading = findParentHeading(state.doc, i);
        if (parentHeading) {
          updateProgressIndicator(view, parentHeading);
        }
        
        return true;
      }
      
      // Stop if we hit a heading
      if (getOrgHeaderLevel(searchLine.text) > 0) {
        break;
      }
    }
    return false;
  }
  
  // Toggle current line checkbox
  const currentState = getCheckboxState(lineText);
  const newState = currentState === 'X' ? ' ' : 'X';
  const newText = setCheckboxState(lineText, newState);
  
  view.dispatch({
    changes: { from: line.from, to: line.to, insert: newText }
  });
  
  // Update parent checkboxes
  updateParentCheckboxStates(view, line.number);
  
  // Update progress indicators
  const parentHeading = findParentHeading(state.doc, line.number);
  if (parentHeading) {
    updateProgressIndicator(view, parentHeading);
  }
  
  return true;
}

function foldAllHeadings(view) {
  const state = view.state;
  const doc = state.doc;
  const effects = [];
  
  // Find all headings and fold them
  for (let i = 1; i <= doc.lines; i++) {
    const line = doc.line(i);
    const level = getOrgHeaderLevel(line.text);
    if (level > 0) {
      const range = findOrgFoldRange(state, i);
      if (range) {
        effects.push(foldEffect.of({ from: range.from, to: range.to }));
      }
    }
  }
  
  if (effects.length > 0) {
    view.dispatch({ effects });
    console.log(`📁 Folded all ${effects.length} headings`);
    return true;
  }
  return false;
}

function unfoldAllHeadings(view) {
  const state = view.state;
  const foldedRanges = state.field(foldState, false);
  
  if (!foldedRanges || foldedRanges.size === 0) {
    return false; // Nothing to unfold
  }
  
  const effects = [];
  const iter = foldedRanges.iter();
  while (iter.value !== null) {
    effects.push(unfoldEffect.of({ from: iter.from, to: iter.to }));
    iter.next();
  }
  
  if (effects.length > 0) {
    view.dispatch({ effects });
    console.log(`📂 Unfolded all ${effects.length} headings`);
    return true;
  }
  return false;
}

function insertNewListItem(view) {
  const state = view.state;
  const pos = state.selection.main.head;
  const line = state.doc.lineAt(pos);
  const lineText = line.text;
  const lineStart = line.from;
  const lineEnd = line.to;
  const cursorInLine = pos - lineStart;
  
  // Check for heading (starts with * at beginning of line, typically no indentation)
  // Use getOrgHeaderLevel to properly detect headings
  const headingLevel = getOrgHeaderLevel(lineText);
  if (headingLevel > 0) {
    // This is a heading - insert another heading at same level
    const level = headingLevel;
    
    // If cursor is in middle of heading, split the line
    if (cursorInLine > 0 && cursorInLine < lineText.length) {
      const beforeCursor = lineText.slice(0, cursorInLine);
      const afterCursor = lineText.slice(cursorInLine);
      
      // Update current line to text before cursor
      const newCurrentLine = beforeCursor.trimEnd();
      // Insert new heading after cursor
      const newHeading = '*'.repeat(level) + ' ' + afterCursor.trimStart();
      
      // Calculate new line end after first change
      const newLineEnd = lineStart + newCurrentLine.length;
      const newCursorPos = newLineEnd + 1 + newHeading.length;
      
      view.dispatch({
        changes: [
          { from: lineStart, to: lineEnd, insert: newCurrentLine },
          { from: newLineEnd, to: newLineEnd, insert: '\n' + newHeading }
        ],
        selection: { anchor: newCursorPos },
        effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
      });
      return true;
    } else {
      // Cursor at end or beginning - insert new heading after
      const newHeading = '*'.repeat(level) + ' ';
      const newCursorPos = lineEnd + 1 + newHeading.length;
      view.dispatch({
        changes: { from: lineEnd, to: lineEnd, insert: '\n' + newHeading },
        selection: { anchor: newCursorPos },
        effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
      });
      return true;
    }
  }
  
  // Check for checkbox list item: - [ ] or - [x] or + [ ] etc. (standard org-mode syntax)
  const checkboxMatch = lineText.match(/^(\s*)([-+*])\s+\[([ xX-])\]\s*(.*)$/);
  if (checkboxMatch) {
    // Standard org-mode checkbox: - [ ] or + [ ] or * [ ]
    const indent = checkboxMatch[1];
    const marker = checkboxMatch[2];
    const checkboxState = checkboxMatch[3];
    const itemText = checkboxMatch[4];
    
    // If cursor is in middle, split the line
    if (cursorInLine > 0 && cursorInLine < lineText.length) {
      const beforeCursor = lineText.slice(0, cursorInLine);
      const afterCursor = lineText.slice(cursorInLine);
      
      // Find where checkbox ends in beforeCursor
      const checkboxEndMatch = beforeCursor.match(/^(\s*[-+*]\s+\[[ xX-]\])\s*/);
      if (checkboxEndMatch) {
        const newCurrentLine = beforeCursor.trimEnd();
        const newCheckbox = indent + marker + ' [ ] ' + afterCursor.trimStart();
        
        // Calculate new line end after first change
        const newLineEnd = lineStart + newCurrentLine.length;
        const newCursorPos = newLineEnd + 1 + newCheckbox.length;
        
        view.dispatch({
          changes: [
            { from: lineStart, to: lineEnd, insert: newCurrentLine },
            { from: newLineEnd, to: newLineEnd, insert: '\n' + newCheckbox }
          ],
          selection: { anchor: newCursorPos },
          effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
        });
        return true;
      }
    }
    
    // Cursor at end or beginning - insert new unchecked checkbox after
    const newCheckbox = indent + marker + ' [ ] ';
    const newCursorPos = lineEnd + 1 + newCheckbox.length;
    view.dispatch({
      changes: { from: lineEnd, to: lineEnd, insert: '\n' + newCheckbox },
      selection: { anchor: newCursorPos },
      effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
    });
    return true;
  }
  
  // Check for numbered list: 1. 2. etc.
  const numberedMatch = lineText.match(/^(\s*)(\d+)\.\s+(.*)$/);
  if (numberedMatch) {
    const indent = numberedMatch[1];
    const currentNumber = parseInt(numberedMatch[2], 10);
    const itemText = numberedMatch[3];
    
    // If cursor is in middle, split the line
    if (cursorInLine > 0 && cursorInLine < lineText.length) {
      const beforeCursor = lineText.slice(0, cursorInLine);
      const afterCursor = lineText.slice(cursorInLine);
      
      // Find where number ends in beforeCursor
      const numberEndMatch = beforeCursor.match(/^(\s*\d+\.)\s*/);
      if (numberEndMatch) {
        const newCurrentLine = beforeCursor.trimEnd();
        const newNumbered = indent + (currentNumber + 1) + '. ' + afterCursor.trimStart();
        
        // Calculate new line end after first change
        const newLineEnd = lineStart + newCurrentLine.length;
        const newCursorPos = newLineEnd + 1 + newNumbered.length;
        
        view.dispatch({
          changes: [
            { from: lineStart, to: lineEnd, insert: newCurrentLine },
            { from: newLineEnd, to: newLineEnd, insert: '\n' + newNumbered }
          ],
          selection: { anchor: newCursorPos },
          effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
        });
        return true;
      }
    }
    
    // Cursor at end or beginning - insert new numbered item after (increment)
    const newNumbered = indent + (currentNumber + 1) + '. ';
    const newCursorPos = lineEnd + 1 + newNumbered.length;
    view.dispatch({
      changes: { from: lineEnd, to: lineEnd, insert: '\n' + newNumbered },
      selection: { anchor: newCursorPos },
      effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
    });
    return true;
  }
  
  // Check for plain bullet list: - + or * (not heading, not checkbox)
  const bulletMatch = lineText.match(/^(\s*)([-+*])\s+(.*)$/);
  if (bulletMatch) {
    const indent = bulletMatch[1];
    const marker = bulletMatch[2];
    const itemText = bulletMatch[3];
    
    // If cursor is in middle, split the line
    if (cursorInLine > 0 && cursorInLine < lineText.length) {
      const beforeCursor = lineText.slice(0, cursorInLine);
      const afterCursor = lineText.slice(cursorInLine);
      
      // Find where bullet marker ends in beforeCursor
      const bulletEndMatch = beforeCursor.match(/^(\s*[-+*])\s*/);
      if (bulletEndMatch) {
        const newCurrentLine = beforeCursor.trimEnd();
        const newBullet = indent + marker + ' ' + afterCursor.trimStart();
        
        // Calculate new line end after first change
        const newLineEnd = lineStart + newCurrentLine.length;
        const newCursorPos = newLineEnd + 1 + newBullet.length;
        
        view.dispatch({
          changes: [
            { from: lineStart, to: lineEnd, insert: newCurrentLine },
            { from: newLineEnd, to: newLineEnd, insert: '\n' + newBullet }
          ],
          selection: { anchor: newCursorPos },
          effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
        });
        return true;
      }
    }
    
    // Cursor at end or beginning - insert new bullet after
    const newBullet = indent + marker + ' ';
    const newCursorPos = lineEnd + 1 + newBullet.length;
    view.dispatch({
      changes: { from: lineEnd, to: lineEnd, insert: '\n' + newBullet },
      selection: { anchor: newCursorPos },
      effects: EditorView.scrollIntoView(newCursorPos, { y: 'start', yMargin: 100 })
    });
    return true;
  }
  
  // Not on a list item - do nothing
  return false;
}

export function createOrgTabKeymap() {
  return [
    {
      key: 'Ctrl-Shift-h',
      run: (view) => {
        const state = view.state;
        const pos = state.selection.main.head;
        const line = state.doc.lineAt(pos);
        
        // If on a header line, toggle fold for that header
        if (getOrgHeaderLevel(line.text) > 0) {
          return toggleFold(view, line.number);
        } else {
          // Find nearest header above and toggle its fold
          for (let i = line.number - 1; i >= 1; i--) {
            if (getOrgHeaderLevel(state.doc.line(i).text) > 0) {
              return toggleFold(view, i);
            }
          }
        }
        return false;
      }
    },
    {
      key: 'Ctrl-Shift-t',
      run: (view) => {
        return toggleCheckboxAtCursor(view);
      }
    },
    {
      key: 'Alt-Enter',
      run: (view) => {
        return insertNewListItem(view);
      }
    },
    {
      key: 'Ctrl-Alt-h',
      run: (view) => {
        return foldAllHeadings(view);
      }
    },
    {
      key: 'Ctrl-Alt-Shift-h',
      run: (view) => {
        return unfoldAllHeadings(view);
      }
    }
  ];
}

function toggleFold(view, lineNum) {
  console.log('🔍 toggleFold called for line:', lineNum);
  const state = view.state;
  const range = findOrgFoldRange(state, lineNum);
  
  if (!range) {
    console.log('❌ No fold range found for line:', lineNum);
    return false;
  }
  
  console.log('✅ Found fold range:', range);
  
  // foldState returns a RangeSet directly, not an object with .ranges
  const foldedRanges = state.field(foldState, false);
  
  // Iterate through all folds to check if our range is folded
  let isFolded = false;
  let foldedRange = null;
  
  if (foldedRanges) {
    const iter = foldedRanges.iter();
    let count = 0;
    while (iter.value !== null) {
      count++;
      console.log('🔍 Checking fold:', { from: iter.from, to: iter.to, targetFrom: range.from, targetTo: range.to });
      if (iter.from === range.from && iter.to === range.to) {
        isFolded = true;
        foldedRange = { from: iter.from, to: iter.to };
        break;
      }
      iter.next();
    }
    console.log('📊 Current fold state:', { hasFoldedRanges: !!foldedRanges, totalFolds: count, isFolded });
  } else {
    console.log('📊 No foldState available');
  }
  
  if (isFolded && foldedRange) {
    console.log('📂 Unfolding range:', foldedRange);
    view.dispatch({ effects: unfoldEffect.of({ from: foldedRange.from, to: foldedRange.to }) });
    return true;
  } else {
    console.log('📁 Folding range:', range);
    view.dispatch({ effects: foldEffect.of({ from: range.from, to: range.to }) });
    return true;
  }
}

// Parse #+STARTUP keywords from org file
function parseStartupKeywords(doc) {
  const startupOptions = {
    overview: false,        // Show only top-level headlines
    content: false,         // Show all headlines
    showall: false,        // Show everything
    show2levels: false,    // Show up to level 2
    show3levels: false,    // Show up to level 3
    show4levels: false,    // Show up to level 4
    show5levels: false,    // Show up to level 5
    hideDrawers: false     // Hide PROPERTIES drawers
  };
  
  // Parse first 50 lines for #+STARTUP keywords
  for (let i = 1; i <= Math.min(50, doc.lines); i++) {
    const line = doc.line(i);
    const text = line.text.trim();
    
    // Match #+STARTUP: option
    const match = text.match(/^#\+STARTUP:\s*(.+)$/i);
    if (match) {
      const options = match[1].toLowerCase().split(/\s+/);
      for (const opt of options) {
        if (opt === 'overview') startupOptions.overview = true;
        else if (opt === 'content') startupOptions.content = true;
        else if (opt === 'showall') startupOptions.showall = true;
        else if (opt === 'show2levels') startupOptions.show2levels = true;
        else if (opt === 'show3levels') startupOptions.show3levels = true;
        else if (opt === 'show4levels') startupOptions.show4levels = true;
        else if (opt === 'show5levels') startupOptions.show5levels = true;
        else if (opt === 'hidedrawers') startupOptions.hideDrawers = true;
      }
    }
  }
  
  return startupOptions;
}

// Parse :VISIBILITY: properties from PROPERTIES drawers
function parseVisibilityProperties(doc) {
  const visibilityMap = new Map(); // line number -> visibility value
  
  let inProperties = false;
  let currentHeadingLine = null;
  let propertiesStartLine = null;
  
  for (let i = 1; i <= doc.lines; i++) {
    const line = doc.line(i);
    const text = line.text.trim();
    
    // Check if this is a heading
    const headingLevel = getOrgHeaderLevel(text);
    if (headingLevel > 0) {
      currentHeadingLine = i;
      inProperties = false;
      continue;
    }
    
    // Check for PROPERTIES drawer start
    if (text === ':PROPERTIES:') {
      inProperties = true;
      propertiesStartLine = currentHeadingLine;
      continue;
    }
    
    // Check for PROPERTIES drawer end
    if (text === ':END:' && inProperties) {
      inProperties = false;
      propertiesStartLine = null;
      continue;
    }
    
    // Parse :VISIBILITY: property
    if (inProperties && propertiesStartLine) {
      const propMatch = text.match(/^:VISIBILITY:\s*(.+)$/i);
      if (propMatch) {
        const visibility = propMatch[1].trim().toLowerCase();
        // Valid values: folded, children, content, all
        if (['folded', 'children', 'content', 'all'].includes(visibility)) {
          visibilityMap.set(propertiesStartLine, visibility);
        }
      }
    }
  }
  
  return visibilityMap;
}

// Apply initial visibility based on #+STARTUP or default
function applyStartupVisibility(state, startupOptions, effects) {
  const doc = state.doc;
  
  if (startupOptions.showall) {
    // Show everything - no folding needed
    return;
  }
  
  if (startupOptions.content) {
    // Show all headlines - no folding needed
    return;
  }
  
  if (startupOptions.overview) {
    // Show only top-level headlines - fold content under each level 1 heading
    for (let i = 1; i <= doc.lines; i++) {
      const line = doc.line(i);
      const level = getOrgHeaderLevel(line.text);
      if (level === 1) {
        // Fold the entire subtree under each level 1 heading
        const range = findOrgFoldRange(state, i);
        if (range) {
          effects.push(foldEffect.of({ from: range.from, to: range.to }));
        }
      }
    }
    return;
  }
  
  // Handle showNlevels
  let maxLevel = 0;
  if (startupOptions.show2levels) maxLevel = 2;
  else if (startupOptions.show3levels) maxLevel = 3;
  else if (startupOptions.show4levels) maxLevel = 4;
  else if (startupOptions.show5levels) maxLevel = 5;
  
  if (maxLevel > 0) {
    for (let i = 1; i <= doc.lines; i++) {
      const line = doc.line(i);
      const level = getOrgHeaderLevel(line.text);
      if (level > maxLevel) {
        // Fold headings deeper than maxLevel
        const range = findOrgFoldRange(state, i);
        if (range) {
          effects.push(foldEffect.of({ from: range.from, to: range.to }));
        }
      }
    }
  }
}

// Fold state persistence - saves and restores which headings are folded
// Supports #+STARTUP keywords and :VISIBILITY: properties
export function createFoldStatePersistencePlugin(documentId, onFoldStateChange) {
  if (!documentId) {
    return ViewPlugin.fromClass(class {
      constructor() {}
      update() {}
    });
  }
  
  return ViewPlugin.fromClass(class {
    constructor(view) {
      this.documentId = documentId;
      this.onFoldStateChange = onFoldStateChange;
      this.isRestoring = false;
      this.hasRestored = false;
      
      // Restore fold state after a short delay to ensure document is fully loaded
      setTimeout(() => {
        this.restoreFoldState(view);
      }, 200);
    }
    
    restoreFoldState(view) {
      if (this.hasRestored) return; // Only restore once
      
      try {
        const doc = view.state.doc;
        const effects = [];
        
        // Check if we've already applied startup settings for this file in this session
        const startupAppliedKey = `org_startup_applied_${this.documentId}`;
        const startupAlreadyApplied = localStorage.getItem(startupAppliedKey) === 'true';
        
        // Get localStorage saved state (for tab switching)
        const storageKey = `org_fold_state_${this.documentId}`;
        const saved = localStorage.getItem(storageKey);
        const savedFoldedLines = saved ? JSON.parse(saved) : [];
        
        // Priority logic:
        // 1. If user has manually folded/unfolded (saved state exists), use that (tab switch)
        // 2. If first load and file has :VISIBILITY: properties, apply those
        // 3. If first load and file has #+STARTUP, apply that
        // 4. Otherwise use saved state or default
        
        const headingsToFold = new Set();
        let source = 'default';
        
        // Check if this is a tab switch (user has saved fold state)
        if (savedFoldedLines.length > 0 && startupAlreadyApplied) {
          // Tab switch - restore user's manual fold state
          for (const lineNum of savedFoldedLines) {
            if (lineNum >= 1 && lineNum <= doc.lines) {
              headingsToFold.add(lineNum);
            }
          }
          source = 'localStorage (tab switch)';
        } else {
          // First load - check file-based settings
          const visibilityMap = parseVisibilityProperties(doc);
          const startupOptions = parseStartupKeywords(doc);
          
          // First priority: :VISIBILITY: properties
          if (visibilityMap.size > 0) {
            for (const [lineNum, visibility] of visibilityMap.entries()) {
              if (visibility === 'folded') {
                headingsToFold.add(lineNum);
              } else if (visibility === 'children') {
                headingsToFold.add(lineNum);
              }
            }
            source = ':VISIBILITY: properties';
            localStorage.setItem(startupAppliedKey, 'true');
          }
          // Second priority: #+STARTUP keywords
          else if (startupOptions.overview || startupOptions.show2levels || 
                   startupOptions.show3levels || startupOptions.show4levels || 
                   startupOptions.show5levels) {
            // Apply startup visibility and collect folds
            const startupEffects = [];
            applyStartupVisibility(view.state, startupOptions, startupEffects);
            
            // Extract line numbers from startup effects
            // We need to find which headings were folded by startup
            if (startupOptions.overview) {
              for (let i = 1; i <= doc.lines; i++) {
                const line = doc.line(i);
                const level = getOrgHeaderLevel(line.text);
                if (level === 1) {
                  headingsToFold.add(i);
                }
              }
            } else {
              // For showNlevels, fold headings deeper than maxLevel
              let maxLevel = 0;
              if (startupOptions.show2levels) maxLevel = 2;
              else if (startupOptions.show3levels) maxLevel = 3;
              else if (startupOptions.show4levels) maxLevel = 4;
              else if (startupOptions.show5levels) maxLevel = 5;
              
              if (maxLevel > 0) {
                for (let i = 1; i <= doc.lines; i++) {
                  const line = doc.line(i);
                  const level = getOrgHeaderLevel(line.text);
                  if (level > maxLevel) {
                    headingsToFold.add(i);
                  }
                }
              }
            }
            source = '#+STARTUP keywords';
            localStorage.setItem(startupAppliedKey, 'true');
          }
          // Third priority: saved state (if exists)
          else if (savedFoldedLines.length > 0) {
            for (const lineNum of savedFoldedLines) {
              if (lineNum >= 1 && lineNum <= doc.lines) {
                headingsToFold.add(lineNum);
              }
            }
            source = 'localStorage';
          }
        }
        
        // Apply all folds
        for (const lineNum of headingsToFold) {
          const range = findOrgFoldRange(view.state, lineNum);
          if (range) {
            effects.push(foldEffect.of({ from: range.from, to: range.to }));
          }
        }
        
        if (effects.length > 0) {
          this.isRestoring = true;
          view.dispatch({ effects });
          console.log(`📂 Restored ${effects.length} folded headings from ${source} for document ${this.documentId}`);
          
          setTimeout(() => {
            this.isRestoring = false;
          }, 150);
        }
        
        this.hasRestored = true;
      } catch (err) {
        console.error('Failed to restore fold state:', err);
        this.isRestoring = false;
        this.hasRestored = true;
      }
    }
    
    saveFoldState(view) {
      if (this.isRestoring) return; // Don't save while restoring
      
      try {
        const doc = view.state.doc;
        const foldedRanges = view.state.field(foldState, false);
        
        if (!foldedRanges) {
          // No folds - clear saved state
          const storageKey = `org_fold_state_${this.documentId}`;
          localStorage.removeItem(storageKey);
          if (this.onFoldStateChange) {
            this.onFoldStateChange([]);
          }
          return;
        }
        
        const foldedLines = [];
        
        // Convert fold ranges to line numbers
        const iter = foldedRanges.iter();
        while (iter.value !== null) {
          // Find which heading line this fold range belongs to
          const foldStart = iter.from;
          
          for (let lineNum = 1; lineNum <= doc.lines; lineNum++) {
            const line = doc.line(lineNum);
            const range = findOrgFoldRange(view.state, lineNum);
            if (range && range.from === foldStart) {
              foldedLines.push(lineNum);
              break;
            }
          }
          
          iter.next();
        }
        
        // Always save to localStorage - this represents user's manual fold state
        // This will be used for tab switching, and will override #+STARTUP after first load
        const storageKey = `org_fold_state_${this.documentId}`;
        localStorage.setItem(storageKey, JSON.stringify(foldedLines));
        
        // Mark that user has interacted with folds (so #+STARTUP won't reapply)
        const startupAppliedKey = `org_startup_applied_${this.documentId}`;
        localStorage.setItem(startupAppliedKey, 'true');
        
        if (this.onFoldStateChange) {
          this.onFoldStateChange(foldedLines);
        }
        
        console.log(`💾 Saved fold state for document ${this.documentId}: ${foldedLines.length} folded headings`);
      } catch (err) {
        console.error('Failed to save fold state:', err);
      }
    }
    
    update(update) {
      // Save fold state when it changes
      if (update.state && !this.isRestoring) {
        const prevFolds = update.startState.field(foldState, false);
        const curFolds = update.state.field(foldState, false);
        
        const prevSize = prevFolds ? prevFolds.size : 0;
        const curSize = curFolds ? curFolds.size : 0;
        
        if (prevSize !== curSize) {
          // Fold state changed - save it
          this.saveFoldState(update.view);
        }
      }
    }
  });
}

// Visual indentation plugin - indents content to match heading level
export function createContentIndentationPlugin(enabled) {
  if (!enabled) {
    return ViewPlugin.fromClass(class {
      constructor() {}
      update() {}
    });
  }
  
  return ViewPlugin.fromClass(class {
    constructor(view) {
      this.decorations = Decoration.none;
      this.updateDecos(view);
    }
    
    update(update) {
      if (update.docChanged || update.viewportChanged) {
        this.updateDecos(update.view);
      }
    }
    
    updateDecos(view) {
      const { doc } = view.state;
      const decos = [];
      
      // Map of line number -> parent heading level
      const headingLevels = new Map();
      let currentHeadingLevel = 0;
      
      for (let i = 1; i <= doc.lines; i++) {
        const line = doc.line(i);
        const lineText = line.text;
        const headingLevel = getOrgHeaderLevel(lineText);
        
        if (headingLevel > 0) {
          currentHeadingLevel = headingLevel;
          // Heading lines themselves should not be indented
          headingLevels.set(i, 0);
        } else {
          // Content line - use current heading level for indentation
          headingLevels.set(i, currentHeadingLevel);
        }
      }
      
      // Apply indentation based on heading level
      for (let i = 1; i <= doc.lines; i++) {
        const line = doc.line(i);
        const level = headingLevels.get(i) || 0;
        
        // Only indent content lines (level > 0 means under a heading)
        if (level > 0) {
          // Calculate padding: 12px per level (matching heading indentation)
          const paddingLeft = level * 12;
          try {
            decos.push(Decoration.line({ 
              class: 'org-content-indented',
              attributes: { style: `padding-left: ${paddingLeft}px;` }
            }).range(line.from));
          } catch (e) {
            // Ignore decoration errors
          }
        }
      }
      
      try {
        this.decorations = Decoration.set(decos, true);
      } catch (e) {
        this.decorations = Decoration.none;
      }
    }
  }, { decorations: v => v.decorations });
}

export const createBaseTheme = (darkMode) => EditorView.baseTheme({
  '&': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-editor': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-scroller': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-content': { 
    fontFamily: 'monospace', 
    fontSize: '14px', 
    lineHeight: '1.5',
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121'
  },
  '.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '&.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-editor.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-gutters': {
    backgroundColor: darkMode ? '#1e1e1e' : '#f5f5f5',
    color: darkMode ? '#858585' : '#999999',
    border: 'none'
  },
  '.cm-activeLineGutter': {
    backgroundColor: darkMode ? '#2d2d2d' : '#e8f2ff'
  },
  '.cm-activeLine': {
    backgroundColor: darkMode ? '#2d2d2d' : '#f0f8ff'
  },
  '.cm-selectionBackground, ::selection': {
    backgroundColor: darkMode ? '#264f78' : '#b3d7ff'
  },
  '.cm-cursor': {
    borderLeftColor: darkMode ? '#ffffff' : '#000000'
  },
  '&.cm-focused .cm-selectionBackground, &.cm-focused ::selection': {
    backgroundColor: darkMode ? '#264f78' : '#b3d7ff'
  },
  '.cm-line': {
    caretColor: darkMode ? '#ffffff' : '#000000'
  },
  '.cm-line.org-heading': { fontWeight: '600' },
  '.cm-line.org-level-1': { fontSize: '18px', paddingTop: '6px', paddingBottom: '2px', paddingLeft: '0px' },
  '.cm-line.org-level-2': { fontSize: '16px', paddingTop: '4px', paddingBottom: '2px', paddingLeft: '12px' },
  '.cm-line.org-level-3': { fontSize: '15px', paddingTop: '2px', paddingBottom: '1px', paddingLeft: '24px' },
  '.cm-line.org-level-4': { fontSize: '14px', paddingLeft: '36px' },
  '.cm-line.org-current-heading': { 
    backgroundColor: darkMode ? '#264f78' : '#e3f2fd', 
    borderLeft: darkMode ? '3px solid #90caf9' : '3px solid #1976d2' 
  },
  '.org-checkbox': { 
    backgroundColor: darkMode ? '#424242' : '#e5e7eb', 
    borderRadius: '2px',
    color: darkMode ? '#b3b3b3' : '#212121'
  },
  '.org-todo-mark': { color: darkMode ? '#f44336' : '#c62828', fontWeight: '700' },
  '.org-done-mark': { color: darkMode ? '#66bb6a' : '#2e7d32', fontWeight: '700' },
  '.org-link': { 
    color: darkMode ? '#90caf9' : '#1976d2', 
    textDecoration: 'underline', 
    cursor: 'pointer' 
  },
  '.org-progress-indicator': {
    color: darkMode ? '#90caf9' : '#1976d2',
    fontWeight: '600',
    marginLeft: '8px'
  },
  '.cm-foldGutter': { width: '16px' },
  '.cm-foldPlaceholder': { 
    backgroundColor: darkMode ? '#424242' : '#eee', 
    border: darkMode ? '1px solid #616161' : '1px solid #ddd', 
    color: darkMode ? '#b3b3b3' : '#888',
    borderRadius: '3px',
    padding: '0 4px',
    fontFamily: 'monospace'
  },
  '.org-content-indented': {
    // Indentation applied via inline style
  }
});
