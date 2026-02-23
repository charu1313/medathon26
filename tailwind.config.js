/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                medical: {
                    blue: '#2563EB',    // Primary Brand
                    teal: '#14B8A6',    // Secondary / Success
                    dark: '#1E293B',    // Headings
                    muted: '#64748B',   // Body text
                    light: '#F8FAFC',   // Background
                    white: '#FFFFFF',
                    border: '#E2E8F0',
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            container: {
                center: true,
                padding: '1.5rem',
                screens: {
                    lg: '1200px',
                },
            },
            boxShadow: {
                'soft': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
                'card': '0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025)',
            }
        },
    },
    plugins: [],
}
