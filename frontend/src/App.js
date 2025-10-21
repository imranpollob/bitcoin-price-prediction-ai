import React from 'react';
import { Container, Navbar, Nav } from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';
import ModelRouter from './components/ModelRouter';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
        <Container>
          <LinkContainer to="/">
            <Navbar.Brand>Bitcoin Price Prediction Dashboard</Navbar.Brand>
          </LinkContainer>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <LinkContainer to="/dashboard">
                <Nav.Link>Dashboard</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/naive">
                <Nav.Link>Naive Model</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/dense">
                <Nav.Link>Dense Model</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/lstm">
                <Nav.Link>LSTM Model</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/conv1d">
                <Nav.Link>Conv1D Model</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/nbeats">
                <Nav.Link>N-BEATS Model</Nav.Link>
              </LinkContainer>
              <LinkContainer to="/ensemble">
                <Nav.Link>Ensemble Model</Nav.Link>
              </LinkContainer>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      
      <Container className="main-container">
        <ModelRouter />
      </Container>
    </div>
  );
}

export default App;