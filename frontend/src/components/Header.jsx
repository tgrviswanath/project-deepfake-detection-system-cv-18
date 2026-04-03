import React from "react";
import { AppBar, Toolbar, Typography } from "@mui/material";
import SecurityIcon from "@mui/icons-material/Security";

export default function Header() {
  return (
    <AppBar position="static" color="primary">
      <Toolbar>
        <SecurityIcon sx={{ mr: 1 }} />
        <Typography variant="h6" fontWeight="bold">
          Deepfake Detection System
        </Typography>
      </Toolbar>
    </AppBar>
  );
}
