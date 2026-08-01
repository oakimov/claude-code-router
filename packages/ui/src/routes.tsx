import { createMemoryRouter, Navigate } from 'react-router';
import App from './App';
import { Login } from '@/components/Login';
import { DebugPage } from '@/components/DebugPage';
import { Presets } from '@/components/Presets';
import ProtectedRoute from '@/components/ProtectedRoute';
import PublicRoute from '@/components/PublicRoute';
import { AppShell } from '@/components/layout/AppShell';

export const router = createMemoryRouter([
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '/login',
    element: <PublicRoute><Login /></PublicRoute>,
  },
  {
    element: <ProtectedRoute><AppShell /></ProtectedRoute>,
    children: [
      {
        path: '/dashboard',
        element: <App />,
      },
      {
        path: '/presets',
        element: <Presets />,
      },
      {
        path: '/debug',
        element: <DebugPage />,
      },
    ],
  },
], {
  initialEntries: ['/dashboard']
});
