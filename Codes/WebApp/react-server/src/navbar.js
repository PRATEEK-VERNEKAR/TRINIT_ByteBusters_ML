import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {


  return (
    <div>
    <nav className="navbar">
      <div className="navbar-left">
        <span className="site-name">EmpowerSafe</span>
      </div>
      <ul className="navbar-right">
        <li><Link to="/">Home</Link></li>
        <li><Link to="/feed">Feed</Link></li>
        <li><Link to="/about">About</Link></li>
      </ul>
    </nav>
    </div>
  );
}

export default Navbar;