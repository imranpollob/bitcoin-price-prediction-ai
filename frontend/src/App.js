import React from 'react';
import { Container, Navbar, Nav } from 'react-bootstrap';
import BitcoinPredictionDashboard from './components/BitcoinPredictionDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
        <Container>
          <Navbar.Brand href="#home">Bitcoin Price Prediction Dashboard</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="#dashboard">Dashboard</Nav.Link>
              <Nav.Link href="#models">Models</Nav.Link>
              <Nav.Link href="#about">About</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      
      <Container className="main-container">
        <BitcoinPredictionDashboard />
      </Container>
    </div>
  );
}

export default App;