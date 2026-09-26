import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'

import App from './App'
import ScreeningPage from './ScreeningPage'
import ResearchPage from './ResearchPage'

import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* Landing page */}
        <Route path="/" element={<App />} />

        {/* Screening */}
        <Route path="/screening" element={<ScreeningPage />} />

        {/* Research */}
        <Route path="/research" element={<ResearchPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)