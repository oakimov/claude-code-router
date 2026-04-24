import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Editor from '@monaco-editor/react';
import { Button } from '@/components/ui/button';
import { api } from '@/lib/api';
import { useTranslation } from 'react-i18next';
import { X, RefreshCw, Download, Trash2, ArrowLeft, File, Layers, Bug } from 'lucide-react';

interface LogViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  showToast?: (message: string, type: 'success' | 'error' | 'warning') => void;
}

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string; // Now this field directly contains the raw JSON string
  source?: string;
  reqId?: string;
  [key: string]: any; // Allow dynamic properties like msg, url, body, etc.
}

interface LogFile {
  name: string;
  path: string;
  size: number;
  lastModified: string;
}

interface GroupedLogs {
  [reqId: string]: LogEntry[];
}

interface LogGroupSummary {
  reqId: string;
  logCount: number;
  firstLog: string;
  lastLog: string;
  model?: string;
}

interface GroupedLogsResponse {
  grouped: boolean;
  groups: { [reqId: string]: LogEntry[] };
  summary: {
    totalRequests: number;
    totalLogs: number;
    requests: LogGroupSummary[];
  };
}

export function LogViewer({ open, onOpenChange, showToast }: LogViewerProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [logs, setLogs] = useState<string[]>([]);
  const [logFiles, setLogFiles] = useState<LogFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<LogFile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [groupByReqId, setGroupByReqId] = useState(false);
  const [groupedLogs, setGroupedLogs] = useState<GroupedLogsResponse | null>(null);
  const [selectedReqId, setSelectedReqId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const refreshInterval = useRef<NodeJS.Timeout | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const editorRef = useRef<any>(null);

  useEffect(() => {
    if (open) {
      loadLogFiles();
    }
  }, [open]);

  // Create inline Web Worker
  const createInlineWorker = (): Worker => {
    const workerCode = `
      // Log aggregation Web Worker
      self.onmessage = function(event) {
        const { type, data } = event.data;
        
        if (type === 'groupLogsByReqId') {
          try {
            const { logs } = data;
            
            // Aggregate logs by reqId
            const groupedLogs = {};
            
            logs.forEach((log, index) => {
              log = JSON.parse(log);
              let reqId = log.reqId || 'no-req-id';
              
              if (!groupedLogs[reqId]) {
                groupedLogs[reqId] = [];
              }
              groupedLogs[reqId].push(log);
            });

            // Sort logs in each group by timestamp
            Object.keys(groupedLogs).forEach(reqId => {
              groupedLogs[reqId].sort((a, b) => a.time - b.time);
            });

            // Extract model information
            const extractModelInfo = (reqId) => {
              const logGroup = groupedLogs[reqId];
              for (const log of logGroup) {
                try {
                  // Try to parse JSON from message field
                  if (log.type === 'request body' && log.data && log.data.model) {
                    return log.data.model;
                  }
                } catch (e) {
                  // Parsing failed, continue to next log line
                }
              }
              return undefined;
            };

            // Generate summary information
            const summary = {
              totalRequests: Object.keys(groupedLogs).length,
              totalLogs: logs.length,
              requests: Object.keys(groupedLogs).map(reqId => ({
                reqId,
                logCount: groupedLogs[reqId].length,
                firstLog: groupedLogs[reqId][0]?.time,
                lastLog: groupedLogs[reqId][groupedLogs[reqId].length - 1]?.time,
                model: extractModelInfo(reqId)
              }))
            };

            const response = {
              grouped: true,
              groups: groupedLogs,
              summary
            };

            // Send results back to main thread
            self.postMessage({
              type: 'groupLogsResult',
              data: response
            });
          } catch (error) {
            // Send error back to main thread
            self.postMessage({
              type: 'error',
              error: error instanceof Error ? error.message : 'Unknown error occurred'
            });
          }
        }
      };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    return new Worker(workerUrl);
  };

  // Initialize Web Worker
  useEffect(() => {
    if (typeof Worker !== 'undefined') {
      try {
        // Create inline Web Worker
        workerRef.current = createInlineWorker();

        // Listen to Worker messages
        workerRef.current.onmessage = (event) => {
          const { type, data, error } = event.data;

          if (type === 'groupLogsResult') {
            setGroupedLogs(data);
          } else if (type === 'error') {
            console.error('Worker error:', error);
            if (showToast) {
              showToast(t('log_viewer.worker_error') + ': ' + error, 'error');
            }
          }
        };

        // Listen to Worker errors
        workerRef.current.onerror = (error) => {
          console.error('Worker error:', error);
          if (showToast) {
            showToast(t('log_viewer.worker_init_failed'), 'error');
          }
        };
      } catch (error) {
        console.error('Failed to create worker:', error);
        if (showToast) {
          showToast(t('log_viewer.worker_init_failed'), 'error');
        }
      }
    }

    // Cleanup Worker
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [showToast, t]);

  useEffect(() => {
    if (autoRefresh && open && selectedFile) {
      refreshInterval.current = setInterval(() => {
        loadLogs();
      }, 5000); // Refresh every 5 seconds
    } else if (refreshInterval.current) {
      clearInterval(refreshInterval.current);
    }

    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    };
  }, [autoRefresh, open, selectedFile]);

  // Load logs when selected file changes
  useEffect(() => {
    if (selectedFile && open) {
      setLogs([]); // Clear existing logs
      loadLogs();
    }
  }, [selectedFile, open]);

  // Handle open/close animations
  useEffect(() => {
    if (open) {
      setIsVisible(true);
      // Trigger the animation after a small delay to ensure the element is rendered
      requestAnimationFrame(() => {
        setIsAnimating(true);
      });
    } else {
      setIsAnimating(false);
      // Wait for the animation to complete before hiding
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [open]);

  const loadLogFiles = async () => {
    try {
      setIsLoading(true);
      const response = await api.getLogFiles();

      if (response && Array.isArray(response)) {
        setLogFiles(response);
        setSelectedFile(null);
        setLogs([]);
      } else {
        setLogFiles([]);
        if (showToast) {
          showToast(t('log_viewer.no_log_files_available'), 'warning');
        }
      }
    } catch (error) {
      console.error('Failed to load log files:', error);
      if (showToast) {
        showToast(t('log_viewer.load_files_failed') + ': ' + (error as Error).message, 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const loadLogs = async () => {
    if (!selectedFile) return;

    try {
      setIsLoading(true);
      setGroupedLogs(null);
      setSelectedReqId(null);

      // Always load raw log data
      const response = await api.getLogs(selectedFile.path);

      if (response && Array.isArray(response)) {
        // Now the interface returns an array of raw log strings, store directly
        setLogs(response);

        // If grouping enabled, use Web Worker for aggregation (needs conversion to LogEntry format for Worker)
        if (groupByReqId && workerRef.current) {
          // const workerLogs: LogEntry[] = response.map((logLine, index) => ({
          //   timestamp: new Date().toISOString(),
          //   level: 'info',
          //   message: logLine,
          //   source: undefined,
          //   reqId: undefined
          // }));

          workerRef.current.postMessage({
            type: 'groupLogsByReqId',
            data: { logs: response }
          });
        } else {
          setGroupedLogs(null);
        }
      } else {
        setLogs([]);
        setGroupedLogs(null);
        if (showToast) {
          showToast(t('log_viewer.no_logs_available'), 'warning');
        }
      }
    } catch (error) {
      console.error('Failed to load logs:', error);
      if (showToast) {
        showToast(t('log_viewer.load_failed') + ': ' + (error as Error).message, 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const clearLogs = async () => {
    if (!selectedFile) return;

    try {
      await api.clearLogs(selectedFile.path);
      setLogs([]);
      if (showToast) {
        showToast(t('log_viewer.logs_cleared'), 'success');
      }
    } catch (error) {
      console.error('Failed to clear logs:', error);
      if (showToast) {
        showToast(t('log_viewer.clear_failed') + ': ' + (error as Error).message, 'error');
      }
    }
  };

  const selectFile = (file: LogFile) => {
    setSelectedFile(file);
    setAutoRefresh(false); // Reset auto refresh when changing files
  };


  const toggleGroupByReqId = () => {
    const newValue = !groupByReqId;
    setGroupByReqId(newValue);

    if (newValue && selectedFile && logs.length > 0) {
      // When grouping enabled and logs already exist, use Worker for aggregation
      if (workerRef.current) {
        workerRef.current.postMessage({
          type: 'groupLogsByReqId',
          data: { logs }
        });
      }
    } else if (!newValue) {
      // When grouping disabled, clear aggregation results
      setGroupedLogs(null);
      setSelectedReqId(null);
    }
  };

  const selectReqId = (reqId: string) => {
    setSelectedReqId(reqId);
  };


  const getDisplayLogs = () => {
    if (groupByReqId && groupedLogs) {
      if (selectedReqId && groupedLogs.groups[selectedReqId]) {
        return groupedLogs.groups[selectedReqId];
      }
      // When in grouping mode but no specific request selected, show raw log string array
      return logs.map(logLine => ({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: logLine,
        source: undefined,
        reqId: undefined
      }));
    }
    // When not in grouping mode, show raw log string array
    return logs.map(logLine => ({
      timestamp: new Date().toISOString(),
      level: 'info',
      message: logLine,
      source: undefined,
      reqId: undefined
    }));
  };

  const downloadLogs = () => {
    if (!selectedFile || logs.length === 0) return;

    // Directly download raw log strings, one log per line
    const logText = logs.join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile.name}-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    if (showToast) {
      showToast(t('log_viewer.logs_downloaded'), 'success');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Breadcrumb navigation item type
  interface BreadcrumbItem {
    id: string;
    label: string;
    onClick: () => void;
  }

  // Get breadcrumb navigation items
  const getBreadcrumbs = (): BreadcrumbItem[] => {
    const breadcrumbs: BreadcrumbItem[] = [
      {
        id: 'root',
        label: t('log_viewer.title'),
        onClick: () => {
          setSelectedFile(null);
          setAutoRefresh(false);
          setLogs([]);
          setGroupedLogs(null);
          setSelectedReqId(null);
          setGroupByReqId(false);
        }
      }
    ];

    if (selectedFile) {
      breadcrumbs.push({
        id: 'file',
        label: selectedFile.name,
        onClick: () => {
          if (groupByReqId) {
            // If in grouping mode, clicking file level should return to group list
            setSelectedReqId(null);
          } else {
            // If not in grouping mode, clicking file level closes grouping feature
            setSelectedReqId(null);
            setGroupedLogs(null);
            setGroupByReqId(false);
          }
        }
      });
    }

    if (selectedReqId) {
      breadcrumbs.push({
        id: 'req',
        label: `${t('log_viewer.request')} ${selectedReqId}`,
        onClick: () => {
          // Do nothing when clicking current level
        }
      });
    }

    return breadcrumbs;
  };

  // Get handler for back button
  const getBackAction = (): (() => void) | null => {
    if (selectedReqId) {
      return () => {
        setSelectedReqId(null);
      };
    } else if (selectedFile) {
      return () => {
        setSelectedFile(null);
        setAutoRefresh(false);
        setLogs([]);
        setGroupedLogs(null);
        setSelectedReqId(null);
        setGroupByReqId(false);
      };
    }
    return null;
  };

  const formatLogsForEditor = () => {
    // If in grouping mode and specific request selected, show logs for that request
    if (groupByReqId && groupedLogs && selectedReqId && groupedLogs.groups[selectedReqId]) {
      const requestLogs = groupedLogs.groups[selectedReqId];
      // Extract raw JSON string and one per line
      return requestLogs.map(log => JSON.stringify(log)).join('\n');
    }

    // Otherwise, directly show raw log string array, one per line
    return logs.join('\n');
  };

  // Parse log line to get line number of final request
  const getFinalRequestLines = () => {
    const lines: number[] = [];

    if (groupByReqId && groupedLogs && selectedReqId && groupedLogs.groups[selectedReqId]) {
      // In grouping mode, check logs of selected request
      const requestLogs = groupedLogs.groups[selectedReqId];
      requestLogs.forEach((log, index) => {
        try {
          // @ts-ignore
          log = JSON.parse(log)
          // Check if the log's msg field equals "final request"
          if (log.msg === "final request") {
            lines.push(index + 1); // Line numbers start from 1
          }
        } catch (e) {
          // Parsing failed, skip
        }
      });
    } else {
      // In non-grouping mode, check raw logs
      logs.forEach((logLine, index) => {
        try {
          const log = JSON.parse(logLine);
          // Check if the log's msg field equals "final request"
          if (log.msg === "final request") {
            lines.push(index + 1); // Line numbers start from 1
          }
        } catch (e) {
          // Parsing failed, skip
        }
      });
    }

    return lines;
  };

  // Handle debug button click
  const handleDebugClick = (lineNumber: number) => {
    console.log('handleDebugClick called with lineNumber:', lineNumber);
    console.log('Current state:', { groupByReqId, selectedReqId, logsLength: logs.length });

    let logData = null;

    if (groupByReqId && groupedLogs && selectedReqId && groupedLogs.groups[selectedReqId]) {
      // Get log data in grouping mode
      const requestLogs = groupedLogs.groups[selectedReqId];
      console.log('Group mode - requestLogs length:', requestLogs.length);
      logData = requestLogs[lineNumber - 1]; // Convert line number to array index
      console.log('Group mode - logData:', logData);
    } else {
      // Get log data in non-grouping mode
      console.log('Non-group mode - logs length:', logs.length);
      try {
        const logLine = logs[lineNumber - 1];
        console.log('Log line:', logLine);
        logData = JSON.parse(logLine);
        console.log('Parsed logData:', logData);
      } catch (e) {
        console.error('Failed to parse log data:', e);
      }
    }

    if (logData) {
      console.log('Navigating to debug page with logData:', logData);
      // Navigate to debug page and pass log data as URL parameter
      const logDataParam = encodeURIComponent(JSON.stringify(logData));
      console.log('Encoded logDataParam length:', logDataParam.length);
      navigate(`/debug?logData=${logDataParam}`);
    } else {
      console.error('No log data found for line:', lineNumber);
    }
  };

  // Configure Monaco Editor
  const configureEditor = (editor: any) => {
    editorRef.current = editor;

    // Enable glyph margin
    editor.updateOptions({
      glyphMargin: true,
    });

    // Store current decoration IDs
    let currentDecorations: string[] = [];

    // Add glyph margin decorations
    const updateDecorations = () => {
      const finalRequestLines = getFinalRequestLines();
      const decorations = finalRequestLines.map(lineNumber => ({
        range: {
          startLineNumber: lineNumber,
          startColumn: 1,
          endLineNumber: lineNumber,
          endColumn: 1
        },
        options: {
          glyphMarginClassName: 'debug-button-glyph',
          glyphMarginHoverMessage: { value: 'Click to debug this request' }
        }
      }));

      // Use deltaDecorations to correctly update decorations and clean up old ones
      currentDecorations = editor.deltaDecorations(currentDecorations, decorations);
    };

    // Initial decorations update
    updateDecorations();

    // Listen to glyph margin clicks - use correct event listener
    editor.onMouseDown((e: any) => {
      console.log('Mouse down event:', e.target);
      console.log('Event details:', {
        type: e.target.type,
        hasDetail: !!e.target.detail,
        glyphMarginLane: e.target.detail?.glyphMarginLane,
        offsetX: e.target.detail?.offsetX,
        glyphMarginLeft: e.target.detail?.glyphMarginLeft,
        glyphMarginWidth: e.target.detail?.glyphMarginWidth
      });

      // Check if click is in glyph margin area
      const isGlyphMarginClick = e.target.detail &&
        e.target.detail.glyphMarginLane !== undefined &&
        e.target.detail.offsetX !== undefined &&
        e.target.detail.offsetX <= e.target.detail.glyphMarginLeft + e.target.detail.glyphMarginWidth;

      console.log('Is glyph margin click:', isGlyphMarginClick);

      if (e.target.position && isGlyphMarginClick) {
        const finalRequestLines = getFinalRequestLines();
        console.log('Final request lines:', finalRequestLines);
        console.log('Clicked line number:', e.target.position.lineNumber);
        if (finalRequestLines.includes(e.target.position.lineNumber)) {
          console.log('Opening debug page for line:', e.target.position.lineNumber);
          handleDebugClick(e.target.position.lineNumber);
        }
      }
    });

    // Try using onGlyphMarginClick if available
    if (typeof editor.onGlyphMarginClick === 'function') {
      editor.onGlyphMarginClick((e: any) => {
        console.log('Glyph margin click event:', e);
        const finalRequestLines = getFinalRequestLines();
        if (finalRequestLines.includes(e.target.position.lineNumber)) {
          console.log('Opening debug page for line (glyph):', e.target.position.lineNumber);
          handleDebugClick(e.target.position.lineNumber);
        }
      });
    }

    // Add mouse move event to detect hover over debug button
    editor.onMouseMove((e: any) => {
      if (e.target.position && (e.target.type === 4 || e.target.type === 'glyph-margin')) {
        const finalRequestLines = getFinalRequestLines();
        if (finalRequestLines.includes(e.target.position.lineNumber)) {
          // Can add hover effect here
          editor.updateOptions({
            glyphMargin: true,
          });
        }
      }
    });

    // Update decorations when logs change
    const interval = setInterval(updateDecorations, 1000);

    return () => {
      clearInterval(interval);
      // Cleanup decorations
      if (editorRef.current) {
        editorRef.current.deltaDecorations(currentDecorations, []);
      }
    };
  };

  if (!isVisible && !open) {
    return null;
  }

  return (
    <>
      {(isVisible || open) && (
        <div
          className={`fixed inset-0 z-50 transition-all duration-300 ease-out ${
            isAnimating && open ? 'bg-black/50 opacity-100' : 'bg-black/0 opacity-0 pointer-events-none'
          }`}
          onClick={() => onOpenChange(false)}
        />
      )}

      <div
        ref={containerRef}
        className={`fixed bottom-0 left-0 right-0 z-50 flex flex-col bg-white shadow-2xl transition-all duration-300 ease-out transform ${
          isAnimating && open ? 'translate-y-0' : 'translate-y-full'
        }`}
        style={{
          height: '100vh',
          maxHeight: '100vh'
        }}
      >
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center gap-2">
            {getBackAction() && (
              <Button
                variant="ghost"
                size="sm"
                onClick={getBackAction()!}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                {t('log_viewer.back')}
              </Button>
            )}

            {/* Breadcrumb navigation */}
            <nav className="flex items-center space-x-1 text-sm">
              {getBreadcrumbs().map((breadcrumb, index) => (
                <React.Fragment key={breadcrumb.id}>
                  {index > 0 && (
                    <span className="text-gray-400 mx-1">/</span>
                  )}
                  {index === getBreadcrumbs().length - 1 ? (
                    <span className="text-gray-900 font-medium">
                      {breadcrumb.label}
                    </span>
                  ) : (
                    <button
                      onClick={breadcrumb.onClick}
                      className="text-blue-600 hover:text-blue-800 transition-colors"
                    >
                      {breadcrumb.label}
                    </button>
                  )}
                </React.Fragment>
              ))}
            </nav>
          </div>
          <div className="flex gap-2">
            {selectedFile && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleGroupByReqId}
                  className={groupByReqId ? 'bg-blue-100 text-blue-700' : ''}
                >
                  <Layers className="h-4 w-4 mr-2" />
                  {groupByReqId ? t('log_viewer.grouped_on') : t('log_viewer.group_by_req_id')}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={autoRefresh ? 'bg-blue-100 text-blue-700' : ''}
                >
                  <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
                  {autoRefresh ? t('log_viewer.auto_refresh_on') : t('log_viewer.auto_refresh_off')}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadLogs}
                  disabled={logs.length === 0}
                >
                  <Download className="h-4 w-4 mr-2" />
                  {t('log_viewer.download')}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearLogs}
                  disabled={logs.length === 0}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  {t('log_viewer.clear')}
                </Button>
              </>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => onOpenChange(false)}
            >
              <X className="h-4 w-4 mr-2" />
              {t('log_viewer.close')}
            </Button>
          </div>
        </div>

        <div className="flex-1 min-h-0 bg-gray-50">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          ) : selectedFile ? (
            <>
              {groupByReqId && groupedLogs && !selectedReqId ? (
                // Show log group list
                <div className="flex flex-col h-full p-6">
                  <div className="mb-4 flex-shrink-0">
                    <h3 className="text-lg font-medium mb-2">{t('log_viewer.request_groups')}</h3>
                    <p className="text-sm text-gray-600">
                      {t('log_viewer.total_requests')}: {groupedLogs.summary.totalRequests} |
                      {t('log_viewer.total_logs')}: {groupedLogs.summary.totalLogs}
                    </p>
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto space-y-3">
                    {groupedLogs.summary.requests.map((request) => (
                      <div
                        key={request.reqId}
                        className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                        onClick={() => selectReqId(request.reqId)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <File className="h-5 w-5 text-blue-600" />
                            <span className="font-medium text-sm">{request.reqId}</span>
                            {request.model && (
                              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                                {request.model}
                              </span>
                            )}
                          </div>
                          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            {request.logCount} {t('log_viewer.logs')}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 space-y-1">
                          <div>{t('log_viewer.first_log')}: {formatDate(request.firstLog)}</div>
                          <div>{t('log_viewer.last_log')}: {formatDate(request.lastLog)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                // Show log content
                <div className="relative h-full">
                  <Editor
                    height="100%"
                    defaultLanguage="json"
                    value={formatLogsForEditor()}
                    theme="vs"
                    options={{
                      minimap: { enabled: true },
                      fontSize: 14,
                      scrollBeyondLastLine: false,
                      automaticLayout: true,
                      wordWrap: 'on',
                      readOnly: true,
                      lineNumbers: 'on',
                      folding: true,
                      renderWhitespace: 'all',
                      glyphMargin: true,
                    }}
                    onMount={configureEditor}
                  />
                </div>
              )}
            </>
          ) : (
            <div className="p-6">
              <h3 className="text-lg font-medium mb-4">{t('log_viewer.select_file')}</h3>
              {logFiles.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  <File className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>{t('log_viewer.no_log_files_available')}</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {logFiles.map((file) => (
                    <div
                      key={file.path}
                      className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                      onClick={() => selectFile(file)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <File className="h-5 w-5 text-blue-600" />
                          <span className="font-medium text-sm">{file.name}</span>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 space-y-1">
                        <div>{formatFileSize(file.size)}</div>
                        <div>{formatDate(file.lastModified)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
