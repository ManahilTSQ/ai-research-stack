if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(function(registrations) {
    if (registrations.length > 0) {
      Promise.all(registrations.map(function(r) { return r.unregister(); })).then(function() {
        window.location.reload();
      });
    }
  });
}
