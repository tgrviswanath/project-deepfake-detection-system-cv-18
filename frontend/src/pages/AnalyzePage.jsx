import React, { useState, useRef } from "react";
import {
  Box, CircularProgress, Alert, Typography, Paper,
  Chip, LinearProgress, Divider,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { analyzeImage } from "../services/deepfakeApi";

export default function AnalyzePage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef();

  const handleFile = async (file) => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await analyzeImage(fd);
      setResult(r.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const verdictColor = result?.verdict === "fake" ? "error" : "success";

  return (
    <Box>
      <Paper
        variant="outlined"
        onClick={() => fileRef.current.click()}
        onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
        onDragOver={(e) => e.preventDefault()}
        sx={{
          p: 3, mb: 2, textAlign: "center", cursor: "pointer", borderStyle: "dashed",
          "&:hover": { bgcolor: "action.hover" },
        }}
      >
        <input ref={fileRef} type="file" hidden accept=".jpg,.jpeg,.png,.bmp,.webp"
          onChange={(e) => handleFile(e.target.files[0])} />
        {loading
          ? <Box>
              <CircularProgress size={28} sx={{ mb: 1 }} />
              <Typography color="text.secondary">Analyzing image…</Typography>
            </Box>
          : <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1 }}>
              <UploadFileIcon color="action" />
              <Typography color="text.secondary">
                Drag & drop or click — JPG / PNG / BMP / WEBP
              </Typography>
            </Box>
        }
      </Paper>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {result && (
        <Box>
          <Box sx={{ display: "flex", gap: 1.5, mb: 2, flexWrap: "wrap", alignItems: "center" }}>
            <Chip
              label={result.verdict.toUpperCase()}
              color={verdictColor}
              size="medium"
              sx={{ fontWeight: "bold", fontSize: 16, px: 1 }}
            />
            <Chip label={`Confidence: ${result.confidence}%`} variant="outlined" size="small" />
            <Chip
              label={result.face_detected ? "Face detected" : "No face — full image used"}
              variant="outlined"
              size="small"
              color={result.face_detected ? "default" : "warning"}
            />
          </Box>

          <Paper variant="outlined" sx={{ p: 1, mb: 2 }}>
            <Box sx={{ display: "flex", gap: 1, mb: 1 }}>
              <Typography variant="body2" sx={{ minWidth: 60 }}>Real</Typography>
              <LinearProgress
                variant="determinate"
                value={result.real_probability}
                color="success"
                sx={{ flex: 1, height: 10, borderRadius: 5, mt: 0.5 }}
              />
              <Typography variant="body2">{result.real_probability}%</Typography>
            </Box>
            <Box sx={{ display: "flex", gap: 1 }}>
              <Typography variant="body2" sx={{ minWidth: 60 }}>Fake</Typography>
              <LinearProgress
                variant="determinate"
                value={result.fake_probability}
                color="error"
                sx={{ flex: 1, height: 10, borderRadius: 5, mt: 0.5 }}
              />
              <Typography variant="body2">{result.fake_probability}%</Typography>
            </Box>
          </Paper>

          <Divider sx={{ mb: 2 }} />
          <Typography variant="subtitle2" gutterBottom>
            {result.face_detected ? "Face Crop Used for Analysis" : "Full Image Used for Analysis"}
          </Typography>
          <Paper variant="outlined" sx={{ p: 1, textAlign: "center" }}>
            <img
              src={`data:image/jpeg;base64,${result.face_crop}`}
              alt="analyzed region"
              style={{ maxWidth: "100%", maxHeight: 320, borderRadius: 4 }}
            />
          </Paper>
        </Box>
      )}
    </Box>
  );
}
