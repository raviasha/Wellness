import { test, expect } from '@playwright/test';

test.describe('Wellness-Outcome E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('frontend loads and displays dashboard', async ({ page }) => {
    await expect(page).toHaveTitle(/Wellness/i);

    // Use more specific locators to avoid strict mode violations
    await expect(page.locator('div').filter({ hasText: /^Resting HR$/ }).first()).toBeVisible();
    await expect(page.locator('div').filter({ hasText: /^HRV$/ }).first()).toBeVisible();
    await expect(page.locator('div').filter({ hasText: /^Body Battery$/ }).first()).toBeVisible();
  });

  test('goal setting interaction', async ({ page }) => {
    await page.locator('#goal-text-input').waitFor({ state: 'visible', timeout: 10000 });
    
    const goalInput = page.locator('#goal-text-input');
    await expect(goalInput).toBeVisible();
    await expect(goalInput).toHaveAttribute('placeholder', /Change your goal/i);

    const setGoalBtn = page.locator('button:has-text("Set Goal")');
    await expect(setGoalBtn).toBeVisible();
  });

  test('navigation to evals tab', async ({ page }) => {
    await page.click('button:has-text("Evals")');
    await page.waitForLoadState('networkidle');
    
    const chartCount = await page.locator('.recharts-responsive-container').count();
    expect(chartCount).toBeGreaterThanOrEqual(1);
  });

  test('navigation to settings', async ({ page }) => {
    await page.locator('button:has-text("Settings")').waitFor({ state: 'visible' });
    await page.click('button:has-text("Settings")');
    await page.waitForLoadState('networkidle');
    
    await expect(page.locator('button:has-text("Add New User")')).toBeVisible({ timeout: 10000 });
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
