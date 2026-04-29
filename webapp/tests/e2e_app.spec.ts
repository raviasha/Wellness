import { test, expect } from '@playwright/test';

test.describe('Wellness-Outcome E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the webapp
    await page.goto('/');
  });

  test('frontend loads and displays dashboard', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/Wellness/i);

    // Verify dashboard metrics are visible
    await expect(page.locator('text=RESTING HR')).toBeVisible();
    await expect(page.locator('text=HRV')).toBeVisible();
    await expect(page.locator('text=BODY BATTERY')).toBeVisible();
  });

  test('goal setting interaction', async ({ page }) => {
    // Check for goal input existence
    const goalInput = page.locator('#goal-text-input');
    await expect(goalInput).toBeVisible();
    await expect(goalInput).toHaveAttribute('placeholder', /Change your goal/i);

    // Check for Set Goal button
    const setGoalBtn = page.locator('button:has-text("Set Goal")');
    await expect(setGoalBtn).toBeVisible();
  });

  test('navigation to evals tab', async ({ page }) => {
    // Click on Evals tab
    await page.click('button:has-text("Evals")');
    
    // Verify evals content (charts)
    await expect(page.locator('.recharts-responsive-container')).toHaveCountAtLeast(1);
  });

  test('navigation to settings', async ({ page }) => {
    // Click on Settings button
    await page.click('button:has-text("Settings")');
    
    // Verify settings content (e.g., Add New User button)
    await expect(page.locator('button:has-text("Add New User")')).toBeVisible();
  });

  test('backend health check', async ({ page }) => {
    const response = await page.request.get('/health');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe('ok');
  });

  test('backend returns 404 for unknown API endpoints', async ({ page }) => {
    const response = await page.request.get('/api/unknown-endpoint');
    expect(response.status()).toBe(404);
  });
});
