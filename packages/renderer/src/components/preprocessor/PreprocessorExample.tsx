import { useState, useEffect } from 'react';
import {
  listPreprocessors,
  downloadPreprocessor,
  runPreprocessor,
  PreprocessorJob,
  type Preprocessor,
} from '@/lib/preprocessor';

/**
 * Example component demonstrating usage of the preprocessor API functions.
 * Shows how to list preprocessors, download them, run them, and track progress.
 */
export function PreprocessorExample() {
  const [preprocessors, setPreprocessors] = useState<Preprocessor[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPreprocessor, setSelectedPreprocessor] = useState<string>('');
  const [inputPath, setInputPath] = useState<string>('');
  const [jobProgress, setJobProgress] = useState<string>('');
  const [jobStatus, setJobStatus] = useState<string>('');
  const [currentJob, setCurrentJob] = useState<PreprocessorJob | null>(null);

  useEffect(() => {
    loadPreprocessors();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentJob) {
        currentJob.disconnect();
      }
    };
  }, [currentJob]);

  const loadPreprocessors = async () => {
    setLoading(true);
    try {
      const result = await listPreprocessors(true);
      if (result.success && result.data) {
        setPreprocessors(result.data.preprocessors);
      } else {
        console.error('Failed to load preprocessors:', result.error);
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Failed to load preprocessors:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (name: string) => {
    try {
      setJobProgress(`Starting download of ${name}...`);
      const result = await downloadPreprocessor(name);
      
      if (result.success && result.data) {
        const job = new PreprocessorJob(result.data.job_id);
        setCurrentJob(job);
        
        // Subscribe to updates
        job.onUpdate((data) => {
          console.log('Download update:', data);
          if (data.progress !== undefined) {
            setJobProgress(`Downloading: ${Math.round(data.progress * 100)}%`);
          }
          if (data.message) {
            setJobProgress(data.message);
          }
        });

        job.onStatus((data) => {
          console.log('Connection status:', data);
          setJobStatus(data.status);
        });

        job.onError((data) => {
          console.error('Download error:', data);
          alert(`Error: ${data.error}`);
        });

        // Connect to WebSocket
        await job.connect();
        setJobProgress(`Download started: ${result.data.job_id}`);
      } else {
        alert(`Failed to start download: ${result.error}`);
      }
    } catch (error) {
      console.error('Download error:', error);
      alert(`Error: ${error}`);
    }
  };

  const handleRun = async () => {
    if (!selectedPreprocessor || !inputPath) {
      alert('Please select a preprocessor and provide an input path');
      return;
    }

    try {
      setJobProgress(`Starting ${selectedPreprocessor}...`);
      const result = await runPreprocessor({
        preprocessor_name: selectedPreprocessor,
        input_path: inputPath,
        download_if_needed: true,
      });

      if (result.success && result.data) {
        const job = new PreprocessorJob(result.data.job_id);
        setCurrentJob(job);

        // Subscribe to updates
        job.onUpdate((data) => {
          console.log('Processing update:', data);
          if (data.status === 'complete') {
            setJobProgress(`Complete! Result: ${data.result_path || 'N/A'}`);
            job.disconnect();
          } else if (data.status === 'error') {
            setJobProgress(`Error: ${data.error || 'Unknown error'}`);
            job.disconnect();
          } else if (data.progress !== undefined) {
            setJobProgress(`Processing: ${Math.round(data.progress * 100)}%`);
          } else if (data.message) {
            setJobProgress(data.message);
          }
        });

        job.onStatus((data) => {
          console.log('Connection status:', data);
          setJobStatus(data.status);
        });

        job.onError((data) => {
          console.error('Processing error:', data);
          setJobProgress(`Error: ${data.error}`);
        });

        // Connect to WebSocket
        await job.connect();
        setJobProgress(`Processing started: ${result.data.job_id}`);
      } else {
        alert(`Failed to start processing: ${result.error}`);
      }
    } catch (error) {
      console.error('Processing error:', error);
      alert(`Error: ${error}`);
    }
  };

  const handleStopJob = async () => {
    if (currentJob) {
      await currentJob.disconnect();
      setCurrentJob(null);
      setJobProgress('Job stopped');
    }
  };

  if (loading) {
    return <div>Loading preprocessors...</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '800px' }}>
      <h2>Preprocessor Example</h2>

      <div style={{ marginBottom: '20px' }}>
        <button onClick={loadPreprocessors}>Refresh Preprocessor List</button>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>Available Preprocessors ({preprocessors.length})</h3>
        <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px' }}>
          {preprocessors.map((prep) => (
            <div key={prep.name} style={{ marginBottom: '10px', padding: '10px', border: '1px solid #eee' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong>{prep.display_name || prep.name}</strong>
                  {prep.category && <span style={{ marginLeft: '10px', color: '#666' }}>({prep.category})</span>}
                  {prep.is_downloaded !== undefined && (
                    <span style={{ marginLeft: '10px', color: prep.is_downloaded ? 'green' : 'orange' }}>
                      {prep.is_downloaded ? '✓ Downloaded' : '⚠ Not Downloaded'}
                    </span>
                  )}
                </div>
                {!prep.is_downloaded && (
                  <button onClick={() => handleDownload(prep.name)}>Download</button>
                )}
              </div>
              {prep.description && <p style={{ margin: '5px 0', fontSize: '0.9em' }}>{prep.description}</p>}
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>Run Preprocessor</h3>
        <div style={{ marginBottom: '10px' }}>
          <label>
            Select Preprocessor:
            <select
              value={selectedPreprocessor}
              onChange={(e) => setSelectedPreprocessor(e.target.value)}
              style={{ marginLeft: '10px', padding: '5px' }}
            >
              <option value="">-- Select --</option>
              {preprocessors
                .filter((p) => p.is_downloaded)
                .map((prep) => (
                  <option key={prep.name} value={prep.name}>
                    {prep.display_name || prep.name}
                  </option>
                ))}
            </select>
          </label>
        </div>
        <div style={{ marginBottom: '10px' }}>
          <label>
            Input Path:
            <input
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="/path/to/input/media"
              style={{ marginLeft: '10px', padding: '5px', width: '300px' }}
            />
          </label>
        </div>
        <button onClick={handleRun} disabled={!selectedPreprocessor || !inputPath}>
          Run Preprocessor
        </button>
        {currentJob && (
          <button onClick={handleStopJob} style={{ marginLeft: '10px' }}>
            Stop Job
          </button>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>Job Status</h3>
        <div style={{ padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
          <div>Connection: {jobStatus || 'Not connected'}</div>
          <div>Progress: {jobProgress || 'No active job'}</div>
        </div>
      </div>
    </div>
  );
}

