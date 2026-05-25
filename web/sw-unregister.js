cat > /home/researcher/ai-research-stack/web/sw-unregister.js << 'EOF'
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(function(registrations) {
    for (let registration of registrations) {
      registration.unregister();
    }
  });
}
EOF
