/**
 * Sports Analytics Dashboard JavaScript
 * Handles real-time data visualization and WebSocket communication
 */

class SportsAnalyticsDashboard {
	constructor() {
		this.ws = null
		this.isConnected = false
		this.frameCount = 0
		this.lastUpdateTime = Date.now()
		this.fpsCounter = 0

		// Chart instances
		this.possessionChart = null
		this.fieldChart = null
		this.heatmapChart = null
		this.timelineChart = null

		// Data storage
		this.gameStats = {}
		this.playerPositions = {}
		this.events = []

		// Simulation state
		this.isSimulating = false
		this.simulationInterval = null

		this.init()
	}

	init() {
		this.connectWebSocket()
		this.initializeCharts()
		this.setupEventListeners()
		this.setupVideoUpload()

		// Start FPS counter
		setInterval(() => this.updateFPS(), 1000)

		// Check processing status periodically
		setInterval(() => this.checkProcessingStatus(), 2000)
	}

	connectWebSocket() {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
		const wsUrl = `${protocol}//${window.location.host}/ws`

		try {
			this.ws = new WebSocket(wsUrl)

			this.ws.onopen = () => {
				console.log('WebSocket connected')
				this.isConnected = true
				this.updateConnectionStatus(true)
			}

			this.ws.onmessage = (event) => {
				try {
					const message = JSON.parse(event.data)
					this.handleMessage(message)
				} catch (error) {
					console.error('Error parsing WebSocket message:', error)
				}
			}

			this.ws.onclose = () => {
				console.log('WebSocket disconnected')
				this.isConnected = false
				this.updateConnectionStatus(false)

				// Attempt to reconnect after 3 seconds
				setTimeout(() => this.connectWebSocket(), 3000)
			}

			this.ws.onerror = (error) => {
				console.error('WebSocket error:', error)
				this.updateConnectionStatus(false)
			}
		} catch (error) {
			console.error('Failed to create WebSocket connection:', error)
			this.updateConnectionStatus(false)
		}
	}

	handleMessage(message) {
		switch (message.type) {
			case 'frame_update':
				this.updateDashboard(message.data, message.stats)
				break
			case 'initial_data':
				this.updateDashboard(message.data, message.stats)
				break
			case 'stats_update':
				this.updateStats(message.stats)
				break
			case 'pong':
				// Handle ping/pong for connection health
				break
			default:
				console.log('Unknown message type:', message.type)
		}
	}

	updateDashboard(frameData, stats) {
		this.frameCount++
		this.fpsCounter++

		// Update frame counter
		document.getElementById(
			'frame-counter'
		).textContent = `Frame: ${this.frameCount}`

		// Update game stats
		this.gameStats = stats
		this.updateGameOverview(stats)
		this.updatePossessionChart(stats)
		this.updateFieldVisualization(frameData)
		this.updateHeatmap(stats)
		this.updateEvents(stats)
		this.updatePerformanceMetrics(stats)
		this.updateTimeline(stats)
		this.updateTeamStats(stats)
		this.updateGameFlow(stats)
		this.updatePlayerStats(stats)
		this.updateEventAnalytics(stats)
	}

	updateGameOverview(stats) {
		// Update player count
		document.getElementById('players-count').textContent =
			stats.players_detected || 0

		// Update ball status
		const ballStatus = document.getElementById('ball-status')
		if (stats.ball_detected) {
			ballStatus.innerHTML = '<i class="fas fa-check text-success"></i>'
		} else {
			ballStatus.innerHTML = '<i class="fas fa-times text-danger"></i>'
		}

		// Update total events count
		const totalEvents = stats.events ? stats.events.length : 0
		this.updateElementText('total-events', totalEvents)
	}

	updatePossessionChart(stats) {
		if (!this.possessionChart) return

		const possessionData = stats.possession_stats || {}
		const total = Object.values(possessionData).reduce(
			(sum, val) => sum + val,
			0
		)

		if (total === 0) return

		const data = [
			{
				values: Object.values(possessionData),
				labels: Object.keys(possessionData).map((key) =>
					key === 'none'
						? 'No Possession'
						: `Team ${key.toUpperCase()}`
				),
				type: 'pie',
				hole: 0.4,
				marker: {
					colors: ['#ff4444', '#4444ff', '#888888'],
				},
			},
		]

		const layout = {
			margin: { t: 0, b: 0, l: 0, r: 0 },
			showlegend: true,
			legend: { orientation: 'h', y: -0.1 },
			font: { size: 10 },
		}

		Plotly.react('possession-chart', data, layout, {
			displayModeBar: false,
		})
	}

	updateFieldVisualization(frameData) {
		if (!frameData || !frameData.objects) return

		const fieldContainer = document.getElementById('field-visualization')

		// Clear existing markers
		fieldContainer
			.querySelectorAll('.player-marker, .ball-marker')
			.forEach((el) => el.remove())

		// Add player markers
		frameData.objects.forEach((obj) => {
			if (obj.type === 'person' && obj.pos_pitch) {
				this.addPlayerMarker(fieldContainer, obj)
			} else if (obj.type === 'sports ball' && obj.pos_pitch) {
				this.addBallMarker(fieldContainer, obj)
			}
		})
	}

	addPlayerMarker(container, player) {
		const marker = document.createElement('div')
		marker.className = `player-marker ${this.getTeamClass(
			player.team_name
		)}`

		// Position on field (assuming field dimensions)
		const fieldWidth = container.offsetWidth
		const fieldHeight = container.offsetHeight
		const x = (player.pos_pitch[0] / 100) * fieldWidth // Assuming 100m field width
		const y = (player.pos_pitch[1] / 60) * fieldHeight // Assuming 60m field height

		marker.style.left = `${Math.max(0, Math.min(x, fieldWidth - 12))}px`
		marker.style.top = `${Math.max(0, Math.min(y, fieldHeight - 12))}px`

		// Add tooltip
		marker.title = `Player ${player.id} - ${
			player.team_name || 'Unknown'
		} - Jersey: ${player.jersey_number || 'N/A'}`

		container.appendChild(marker)
	}

	addBallMarker(container, ball) {
		const marker = document.createElement('div')
		marker.className = 'ball-marker'

		const fieldWidth = container.offsetWidth
		const fieldHeight = container.offsetHeight
		const x = (ball.pos_pitch[0] / 100) * fieldWidth
		const y = (ball.pos_pitch[1] / 60) * fieldHeight

		marker.style.left = `${Math.max(0, Math.min(x, fieldWidth - 8))}px`
		marker.style.top = `${Math.max(0, Math.min(y, fieldHeight - 8))}px`

		marker.title = 'Ball'

		container.appendChild(marker)
	}

	getTeamClass(teamName) {
		if (!teamName || teamName === 'unknown' || teamName === 'none') {
			return 'unknown'
		}
		return teamName.toLowerCase().includes('a') ? 'team-a' : 'team-b'
	}

	updateHeatmap(stats) {
		// Simple heatmap visualization
		// In a real implementation, you'd create a proper heatmap
		if (!this.heatmapChart) {
			this.initializeHeatmap()
		}

		// Update with player position data
		const positions = stats.player_positions || {}
		const heatmapData = this.generateHeatmapData(positions)

		if (heatmapData.length > 0) {
			Plotly.react(
				'heatmap-chart',
				heatmapData,
				{
					margin: { t: 0, b: 0, l: 0, r: 0 },
					xaxis: { visible: false },
					yaxis: { visible: false },
				},
				{ displayModeBar: false }
			)
		}
	}

	generateHeatmapData(positions) {
		// Generate heatmap data from player positions
		const data = []

		Object.values(positions).forEach((playerPositions) => {
			if (playerPositions.length > 0) {
				const x = playerPositions.map((pos) => pos.x)
				const y = playerPositions.map((pos) => pos.y)

				data.push({
					x: x,
					y: y,
					type: 'scatter',
					mode: 'markers',
					marker: {
						size: 4,
						opacity: 0.6,
						color: 'red',
					},
				})
			}
		})

		return data
	}

	updateEvents(stats) {
		const eventsContainer = document.getElementById('events-list')
		const events = stats.events || []

		if (events.length === 0) {
			eventsContainer.innerHTML =
				'<div class="text-muted text-center p-3">No events yet...</div>'
			return
		}

		// Show only the latest 10 events
		const recentEvents = events.slice(-10).reverse()

		eventsContainer.innerHTML = recentEvents
			.map(
				(event) => `
            <div class="event-item">
                <div class="event-time">${this.formatTimestamp(
					event.timestamp
				)}</div>
                <div class="event-description">
                    Player ${event.player_id} - ${event.team_name}
                </div>
                <span class="event-type ${event.event_type.toLowerCase()}">${
					event.event_type
				}</span>
            </div>
        `
			)
			.join('')
	}

	updatePerformanceMetrics(stats) {
		const metrics = stats.performance_metrics || {}

		document.getElementById('processing-fps').textContent = Math.round(
			metrics.fps || 0
		)
		document.getElementById('processing-time').textContent = `${Math.round(
			metrics.processing_time || 0
		)}ms`
	}

	updateTimeline(stats) {
		// Simple timeline chart
		const events = stats.events || []

		if (events.length === 0) return

		const timelineData = [
			{
				x: events.map((event) => event.timestamp),
				y: events.map((event, i) => i),
				text: events.map(
					(event) => `${event.event_type} - Player ${event.player_id}`
				),
				mode: 'markers+lines',
				type: 'scatter',
				marker: { size: 8 },
			},
		]

		const layout = {
			margin: { t: 10, b: 30, l: 30, r: 10 },
			xaxis: { title: 'Time (s)' },
			yaxis: { title: 'Events' },
			showlegend: false,
		}

		Plotly.react('timeline-chart', timelineData, layout, {
			displayModeBar: false,
		})
	}

	updateTeamStats(stats) {
		// Update team performance statistics
		const teamStats = stats.team_stats || {}

		// Update Team A stats
		if (teamStats.team_A) {
			const teamA = teamStats.team_A
			this.updateTeamStatsDisplay('team-a', teamA, 'Team A')
		}

		// Update Team B stats
		if (teamStats.team_B) {
			const teamB = teamStats.team_B
			this.updateTeamStatsDisplay('team-b', teamB, 'Team B')
		}

		// Update pass completion rates chart
		this.updatePassCompletionChart(teamStats)
	}

	updateTeamStatsDisplay(elementPrefix, teamData, teamName) {
		// Update individual team stat elements
		const passAccuracy =
			teamData.total_passes > 0
				? (
						(teamData.successful_passes / teamData.total_passes) *
						100
				  ).toFixed(1)
				: 0

		const shotAccuracy =
			teamData.shots_taken > 0
				? (
						(teamData.shots_on_goal / teamData.shots_taken) *
						100
				  ).toFixed(1)
				: 0

		// Update DOM elements if they exist
		this.updateElementText(
			`${elementPrefix}-passes`,
			`${teamData.successful_passes}/${teamData.total_passes}`
		)
		this.updateElementText(
			`${elementPrefix}-pass-accuracy`,
			`${passAccuracy}%`
		)
		this.updateElementText(`${elementPrefix}-shots`, teamData.shots_taken)
		this.updateElementText(
			`${elementPrefix}-shots-on-goal`,
			teamData.shots_on_goal
		)
		this.updateElementText(
			`${elementPrefix}-shot-accuracy`,
			`${shotAccuracy}%`
		)
		this.updateElementText(`${elementPrefix}-goals`, teamData.goals_scored)
		this.updateElementText(
			`${elementPrefix}-distance`,
			`${(teamData.distance_covered / 1000).toFixed(1)}km`
		)
		this.updateElementText(
			`${elementPrefix}-avg-speed`,
			`${teamData.average_speed.toFixed(1)} km/h`
		)
		this.updateElementText(
			`${elementPrefix}-ball-touches`,
			teamData.ball_touches
		)
		this.updateElementText(
			`${elementPrefix}-defensive-actions`,
			teamData.defensive_actions
		)
	}

	updateElementText(elementId, text) {
		const element = document.getElementById(elementId)
		if (element) {
			element.textContent = text
		}
	}

	updatePassCompletionChart(teamStats) {
		const chartElement = document.getElementById('pass-completion-chart')
		if (!chartElement) return

		const teamA = teamStats.team_A || {}
		const teamB = teamStats.team_B || {}

		const teamAAccuracy =
			teamA.total_passes > 0
				? (teamA.successful_passes / teamA.total_passes) * 100
				: 0
		const teamBAccuracy =
			teamB.total_passes > 0
				? (teamB.successful_passes / teamB.total_passes) * 100
				: 0

		const data = [
			{
				x: ['Team A', 'Team B'],
				y: [teamAAccuracy, teamBAccuracy],
				type: 'bar',
				marker: {
					color: ['#ff4444', '#4444ff'],
				},
			},
		]

		const layout = {
			margin: { t: 20, b: 40, l: 40, r: 20 },
			yaxis: { title: 'Pass Accuracy (%)', range: [0, 100] },
			showlegend: false,
			font: { size: 10 },
		}

		Plotly.react('pass-completion-chart', data, layout, {
			displayModeBar: false,
		})
	}

	updateGameFlow(stats) {
		// Update game flow indicators
		const gameFlow = stats.game_flow || {}

		// Update momentum indicator
		this.updateMomentumIndicator(gameFlow.momentum_indicator || 0)

		// Update game intensity
		this.updateElementText(
			'game-intensity',
			(gameFlow.game_intensity || 0).toFixed(1)
		)

		// Update possession changes chart
		this.updatePossessionChangesChart(gameFlow.possession_changes || [])
	}

	updateMomentumIndicator(momentum) {
		const indicator = document.getElementById('momentum-indicator')
		if (!indicator) return

		// Convert momentum (-1 to 1) to percentage (0 to 100)
		const percentage = ((momentum + 1) / 2) * 100

		// Update progress bar
		const progressBar = indicator.querySelector('.progress-bar')
		if (progressBar) {
			progressBar.style.width = `${percentage}%`

			// Change color based on momentum
			if (momentum < -0.3) {
				progressBar.className = 'progress-bar bg-danger'
			} else if (momentum > 0.3) {
				progressBar.className = 'progress-bar bg-primary'
			} else {
				progressBar.className = 'progress-bar bg-warning'
			}
		}
	}

	updatePossessionChangesChart(possessionChanges) {
		const chartElement = document.getElementById('possession-changes-chart')
		if (!chartElement || possessionChanges.length === 0) return

		const data = [
			{
				x: possessionChanges.map((change) => change.timestamp),
				y: possessionChanges.map((change) =>
					change.team === 'team_A' ? 1 : 0
				),
				mode: 'lines+markers',
				type: 'scatter',
				line: { shape: 'hv' },
				marker: { size: 6 },
				name: 'Possession',
			},
		]

		const layout = {
			margin: { t: 20, b: 40, l: 40, r: 20 },
			xaxis: { title: 'Time (s)' },
			yaxis: {
				title: 'Team',
				tickvals: [0, 1],
				ticktext: ['Team B', 'Team A'],
			},
			showlegend: false,
			font: { size: 10 },
		}

		Plotly.react('possession-changes-chart', data, layout, {
			displayModeBar: false,
		})
	}

	updatePlayerStats(stats) {
		// Update individual player statistics
		const playerStats = stats.player_stats || {}

		// Update player stats table
		this.updatePlayerStatsTable(playerStats)

		// Update player performance charts
		this.updatePlayerDistanceChart(playerStats)
		this.updatePlayerSpeedChart(playerStats)
	}

	updatePlayerStatsTable(playerStats) {
		const tbody = document.getElementById('player-stats-tbody')
		if (!tbody) return

		// Clear existing rows
		tbody.innerHTML = ''

		const players = Object.entries(playerStats)

		if (players.length === 0) {
			tbody.innerHTML =
				'<tr><td colspan="9" class="text-center text-muted">No player data available</td></tr>'
			return
		}

		// Sort players by distance covered (descending)
		players.sort(
			(a, b) =>
				(b[1].distance_covered || 0) - (a[1].distance_covered || 0)
		)

		players.forEach(([playerId, playerData]) => {
			const row = document.createElement('tr')

			// Format team name
			let teamName = playerData.team || 'Unknown'
			if (teamName.includes('Team_A') || teamName.includes('Cluster 0')) {
				teamName = 'Team A'
			} else if (
				teamName.includes('Team_B') ||
				teamName.includes('Cluster 1')
			) {
				teamName = 'Team B'
			}

			row.innerHTML = `
				<td><strong>${playerId}</strong></td>
				<td><span class="badge ${
					teamName === 'Team A' ? 'bg-danger' : 'bg-primary'
				}">${teamName}</span></td>
				<td>${(playerData.distance_covered / 1000).toFixed(2)}</td>
				<td>${(playerData.average_speed || 0).toFixed(1)}</td>
				<td>${(playerData.max_speed || 0).toFixed(1)}</td>
				<td>${playerData.ball_touches || 0}</td>
				<td>${playerData.passes_made || 0}</td>
				<td>${playerData.shots_taken || 0}</td>
				<td>${(playerData.possession_time || 0).toFixed(1)}</td>
			`

			tbody.appendChild(row)
		})
	}

	updatePlayerDistanceChart(playerStats) {
		const chartElement = document.getElementById('player-distance-chart')
		if (!chartElement) return

		const players = Object.entries(playerStats)
		if (players.length === 0) return

		// Sort by distance and take top 10
		players.sort(
			(a, b) =>
				(b[1].distance_covered || 0) - (a[1].distance_covered || 0)
		)
		const topPlayers = players.slice(0, 10)

		const data = [
			{
				x: topPlayers.map(([playerId]) => `Player ${playerId}`),
				y: topPlayers.map(([, playerData]) =>
					(playerData.distance_covered / 1000).toFixed(2)
				),
				type: 'bar',
				marker: {
					color: topPlayers.map(([, playerData]) => {
						const team = playerData.team || ''
						if (
							team.includes('Team_A') ||
							team.includes('Cluster 0')
						) {
							return '#ff4444'
						} else if (
							team.includes('Team_B') ||
							team.includes('Cluster 1')
						) {
							return '#4444ff'
						}
						return '#888888'
					}),
				},
			},
		]

		const layout = {
			margin: { t: 20, b: 60, l: 50, r: 20 },
			xaxis: { title: 'Players' },
			yaxis: { title: 'Distance (km)' },
			showlegend: false,
			font: { size: 10 },
		}

		Plotly.react('player-distance-chart', data, layout, {
			displayModeBar: false,
		})
	}

	updatePlayerSpeedChart(playerStats) {
		const chartElement = document.getElementById('player-speed-chart')
		if (!chartElement) return

		const players = Object.entries(playerStats)
		if (players.length === 0) return

		// Prepare data for speed distribution
		const teamAPlayers = []
		const teamBPlayers = []

		players.forEach(([playerId, playerData]) => {
			const avgSpeed = playerData.average_speed || 0
			const team = playerData.team || ''

			if (team.includes('Team_A') || team.includes('Cluster 0')) {
				teamAPlayers.push(avgSpeed)
			} else if (team.includes('Team_B') || team.includes('Cluster 1')) {
				teamBPlayers.push(avgSpeed)
			}
		})

		const data = []

		if (teamAPlayers.length > 0) {
			data.push({
				x: teamAPlayers,
				type: 'histogram',
				name: 'Team A',
				marker: { color: '#ff4444', opacity: 0.7 },
				nbinsx: 10,
			})
		}

		if (teamBPlayers.length > 0) {
			data.push({
				x: teamBPlayers,
				type: 'histogram',
				name: 'Team B',
				marker: { color: '#4444ff', opacity: 0.7 },
				nbinsx: 10,
			})
		}

		const layout = {
			margin: { t: 20, b: 40, l: 50, r: 20 },
			xaxis: { title: 'Average Speed (km/h)' },
			yaxis: { title: 'Number of Players' },
			barmode: 'overlay',
			showlegend: true,
			legend: { orientation: 'h', y: -0.2 },
			font: { size: 10 },
		}

		if (data.length > 0) {
			Plotly.react('player-speed-chart', data, layout, {
				displayModeBar: false,
			})
		}
	}

	updateEventAnalytics(stats) {
		// Update advanced event analytics
		const eventAnalytics = stats.event_analytics || {}

		// Update event frequency chart
		this.updateEventFrequencyChart(eventAnalytics.event_frequency || {})

		// Update success rates
		this.updateSuccessRates(eventAnalytics.event_success_rates || {})

		// Update heat zones chart
		this.updateHeatZonesChart(eventAnalytics.heat_zones || {})
	}

	updateEventFrequencyChart(eventFrequency) {
		const chartElement = document.getElementById('event-frequency-chart')
		if (!chartElement) return

		const events = Object.entries(eventFrequency)
		if (events.length === 0) return

		const data = [
			{
				labels: events.map(([event]) => event),
				values: events.map(([, count]) => count),
				type: 'pie',
				hole: 0.4,
				marker: {
					colors: [
						'#28a745',
						'#dc3545',
						'#ffc107',
						'#17a2b8',
						'#6f42c1',
						'#fd7e14',
					],
				},
			},
		]

		const layout = {
			margin: { t: 20, b: 20, l: 20, r: 20 },
			showlegend: true,
			legend: { orientation: 'v', x: 1.05, y: 0.5 },
			font: { size: 10 },
		}

		Plotly.react('event-frequency-chart', data, layout, {
			displayModeBar: false,
		})
	}

	updateSuccessRates(successRates) {
		// Update pass success rate
		const passData = successRates.Pass || { successful: 0, total: 0 }
		const passRate =
			passData.total > 0
				? (passData.successful / passData.total) * 100
				: 0
		this.updateElementText('pass-success-rate', `${passRate.toFixed(1)}%`)
		this.updateProgressBar('pass-progress', passRate)

		// Update shot success rate
		const shotData = successRates.Shot || { successful: 0, total: 0 }
		const shotRate =
			shotData.total > 0
				? (shotData.successful / shotData.total) * 100
				: 0
		this.updateElementText('shot-success-rate', `${shotRate.toFixed(1)}%`)
		this.updateProgressBar('shot-progress', shotRate)

		// Update dribble success rate
		const dribbleData = successRates.Dribble || { successful: 0, total: 0 }
		const dribbleRate =
			dribbleData.total > 0
				? (dribbleData.successful / dribbleData.total) * 100
				: 0
		this.updateElementText(
			'dribble-success-rate',
			`${dribbleRate.toFixed(1)}%`
		)
		this.updateProgressBar('dribble-progress', dribbleRate)
	}

	updateProgressBar(elementId, percentage) {
		const progressBar = document.getElementById(elementId)
		if (progressBar) {
			progressBar.style.width = `${Math.min(percentage, 100)}%`
		}
	}

	updateHeatZonesChart(heatZones) {
		const chartElement = document.getElementById('heat-zones-chart')
		if (!chartElement) return

		const zones = ['defensive_third', 'middle_third', 'attacking_third']
		const teamAData = zones.map((zone) => heatZones[zone]?.team_A || 0)
		const teamBData = zones.map((zone) => heatZones[zone]?.team_B || 0)

		const data = [
			{
				x: zones.map((zone) =>
					zone
						.replace('_', ' ')
						.replace(/\b\w/g, (l) => l.toUpperCase())
				),
				y: teamAData,
				name: 'Team A',
				type: 'bar',
				marker: { color: '#ff4444' },
			},
			{
				x: zones.map((zone) =>
					zone
						.replace('_', ' ')
						.replace(/\b\w/g, (l) => l.toUpperCase())
				),
				y: teamBData,
				name: 'Team B',
				type: 'bar',
				marker: { color: '#4444ff' },
			},
		]

		const layout = {
			margin: { t: 20, b: 60, l: 40, r: 20 },
			xaxis: { title: 'Field Zones' },
			yaxis: { title: 'Activity Count' },
			barmode: 'group',
			showlegend: true,
			legend: { orientation: 'h', y: -0.3 },
			font: { size: 10 },
		}

		Plotly.react('heat-zones-chart', data, layout, {
			displayModeBar: false,
		})
	}

	initializeCharts() {
		// Initialize empty charts
		this.initializePossessionChart()
		this.initializeHeatmap()
	}

	initializePossessionChart() {
		const data = [
			{
				values: [1],
				labels: ['No Data'],
				type: 'pie',
				hole: 0.4,
				marker: { colors: ['#e9ecef'] },
			},
		]

		const layout = {
			margin: { t: 0, b: 0, l: 0, r: 0 },
			showlegend: false,
			font: { size: 10 },
		}

		Plotly.newPlot('possession-chart', data, layout, {
			displayModeBar: false,
		})
		this.possessionChart = true
	}

	initializeHeatmap() {
		const data = [
			{
				x: [0],
				y: [0],
				type: 'scatter',
				mode: 'markers',
				marker: { size: 1, opacity: 0 },
			},
		]

		Plotly.newPlot(
			'heatmap-chart',
			data,
			{
				margin: { t: 0, b: 0, l: 0, r: 0 },
				xaxis: { visible: false },
				yaxis: { visible: false },
			},
			{ displayModeBar: false }
		)
		this.heatmapChart = true
	}

	setupEventListeners() {
		// Handle window resize
		window.addEventListener('resize', () => {
			this.resizeCharts()
		})

		// Handle simulation button
		const simulationBtn = document.getElementById('simulation-btn')
		if (simulationBtn) {
			simulationBtn.addEventListener('click', () => {
				this.toggleSimulation()
			})
		}
	}

	setupVideoUpload() {
		const fileInput = document.getElementById('video-file')
		const selectUploadBtn = document.getElementById('select-upload-btn')
		const startBtn = document.getElementById('start-analysis-btn')
		const stopBtn = document.getElementById('stop-analysis-btn')

		// Handle single button click - opens file dialog
		selectUploadBtn.addEventListener('click', () => {
			fileInput.click()
		})

		// Auto-upload when file is selected
		fileInput.addEventListener('change', (e) => {
			if (e.target.files.length > 0) {
				this.showSelectedFileInfo(e.target.files[0])
				this.uploadVideo()
			}
		})

		// Handle analysis start/stop
		startBtn.addEventListener('click', () => this.startAnalysis())
		stopBtn.addEventListener('click', () => this.stopAnalysis())
	}

	showSelectedFileInfo(file) {
		const fileInfoDiv = document.getElementById('selected-file-info')
		const fileNameElement = document.getElementById('file-name')
		const fileSizeElement = document.getElementById('file-size')

		fileNameElement.textContent = file.name
		fileSizeElement.textContent = `Size: ${(
			file.size /
			(1024 * 1024)
		).toFixed(2)} MB`
		fileInfoDiv.style.display = 'block'
	}

	async uploadVideo() {
		const fileInput = document.getElementById('video-file')
		const selectUploadBtn = document.getElementById('select-upload-btn')
		const progressBar = document.getElementById('upload-progress')

		if (!fileInput.files.length) {
			this.showUploadStatus('Please select a video file', 'danger')
			return
		}

		const file = fileInput.files[0]
		const formData = new FormData()
		formData.append('file', file)

		try {
			// Update button state
			selectUploadBtn.disabled = true
			selectUploadBtn.innerHTML =
				'<i class="fas fa-spinner fa-spin me-2"></i>Uploading...'

			progressBar.style.display = 'block'
			this.showUploadStatus('Uploading video...', 'info')

			const response = await fetch('/api/upload-video', {
				method: 'POST',
				body: formData,
			})

			const result = await response.json()

			if (response.ok) {
				this.showUploadStatus(result.message, 'success')
				document.getElementById('start-analysis-btn').disabled = false
				progressBar.style.display = 'none'

				// Update button to show success
				selectUploadBtn.innerHTML =
					'<i class="fas fa-check me-2"></i>Video Uploaded'
				selectUploadBtn.className = 'btn btn-success btn-lg'
			} else {
				throw new Error(result.detail || 'Upload failed')
			}
		} catch (error) {
			this.showUploadStatus(`Upload failed: ${error.message}`, 'danger')
			progressBar.style.display = 'none'

			// Reset button on error
			selectUploadBtn.innerHTML =
				'<i class="fas fa-file-video me-2"></i>Upload Video'
			selectUploadBtn.className = 'btn btn-primary btn-lg'
		} finally {
			selectUploadBtn.disabled = false
		}
	}

	async startAnalysis() {
		try {
			const response = await fetch('/api/start-analysis', {
				method: 'POST',
			})

			const result = await response.json()

			if (response.ok) {
				this.showUploadStatus(
					'Analysis started successfully',
					'success'
				)
				this.updateProcessingStatus(true)
			} else {
				throw new Error(result.detail || 'Failed to start analysis')
			}
		} catch (error) {
			this.showUploadStatus(
				`Failed to start analysis: ${error.message}`,
				'danger'
			)
		}
	}

	async stopAnalysis() {
		try {
			const response = await fetch('/api/stop-analysis', {
				method: 'POST',
			})

			const result = await response.json()

			if (response.ok) {
				this.showUploadStatus('Analysis stopped', 'warning')
				this.updateProcessingStatus(false)
			} else {
				throw new Error(result.detail || 'Failed to stop analysis')
			}
		} catch (error) {
			this.showUploadStatus(
				`Failed to stop analysis: ${error.message}`,
				'danger'
			)
		}
	}

	async checkProcessingStatus() {
		try {
			const response = await fetch('/api/status')
			const status = await response.json()

			this.updateProcessingStatus(
				status.is_processing,
				status.current_video
			)
		} catch (error) {
			console.error('Failed to check processing status:', error)
		}
	}

	updateProcessingStatus(isProcessing, currentVideo = null) {
		const statusDiv = document.getElementById('processing-status')
		const startBtn = document.getElementById('start-analysis-btn')
		const stopBtn = document.getElementById('stop-analysis-btn')
		const selectUploadBtn = document.getElementById('select-upload-btn')

		if (isProcessing) {
			statusDiv.innerHTML =
				'<i class="fas fa-circle text-success me-2"></i><span>Processing</span>'
			startBtn.disabled = true
			stopBtn.disabled = false
			selectUploadBtn.disabled = true
		} else {
			statusDiv.innerHTML =
				'<i class="fas fa-circle text-secondary me-2"></i><span>Ready</span>'
			startBtn.disabled = !currentVideo
			stopBtn.disabled = true

			// Reset upload button if not processing and no video uploaded
			if (!currentVideo) {
				selectUploadBtn.disabled = false
				selectUploadBtn.innerHTML =
					'<i class="fas fa-file-video me-2"></i>Upload Video'
				selectUploadBtn.className = 'btn btn-primary btn-lg'
			}
		}
	}

	showUploadStatus(message, type) {
		const statusDiv = document.getElementById('upload-status')
		statusDiv.className = `alert alert-${type}`
		statusDiv.textContent = message
		statusDiv.style.display = 'block'

		// Auto-hide success messages after 3 seconds
		if (type === 'success') {
			setTimeout(() => {
				statusDiv.style.display = 'none'
			}, 3000)
		}
	}

	resizeCharts() {
		// Resize all Plotly charts
		Plotly.Plots.resize('possession-chart')
		Plotly.Plots.resize('heatmap-chart')
		Plotly.Plots.resize('timeline-chart')
	}

	updateConnectionStatus(connected) {
		const statusElement = document.getElementById('connection-status')
		if (connected) {
			statusElement.innerHTML =
				'<i class="fas fa-circle me-1"></i>Connected'
			statusElement.className = 'badge bg-success me-3'
		} else {
			statusElement.innerHTML =
				'<i class="fas fa-circle me-1"></i>Disconnected'
			statusElement.className = 'badge bg-danger me-3'
		}
	}

	updateFPS() {
		document.getElementById(
			'fps-counter'
		).textContent = `FPS: ${this.fpsCounter}`
		this.fpsCounter = 0
	}

	formatTimestamp(timestamp) {
		const minutes = Math.floor(timestamp / 60)
		const seconds = Math.floor(timestamp % 60)
		return `${minutes.toString().padStart(2, '0')}:${seconds
			.toString()
			.padStart(2, '0')}`
	}

	toggleSimulation() {
		const simulationBtn = document.getElementById('simulation-btn')

		if (this.isSimulating) {
			// Stop simulation
			this.stopSimulation()
			simulationBtn.innerHTML =
				'<i class="fas fa-play-circle me-2"></i>Simulation'
			simulationBtn.className = 'btn btn-warning'
		} else {
			// Start simulation
			this.startSimulation()
			simulationBtn.innerHTML =
				'<i class="fas fa-stop-circle me-2"></i>Stop Simulation'
			simulationBtn.className = 'btn btn-danger'
		}
	}

	startSimulation() {
		if (this.isSimulating) return

		this.isSimulating = true

		// Send start simulation message to backend
		if (this.ws && this.isConnected) {
			this.ws.send(
				JSON.stringify({
					type: 'start_simulation',
				})
			)
		}

		// Update processing status
		const statusElement = document.getElementById('processing-status')
		if (statusElement) {
			statusElement.innerHTML =
				'<i class="fas fa-circle text-warning me-2"></i><span>Simulation Running</span>'
		}
	}

	stopSimulation() {
		if (!this.isSimulating) return

		this.isSimulating = false
		console.log('Stopping simulation mode...')

		// Send stop simulation message to backend
		if (this.ws && this.isConnected) {
			this.ws.send(
				JSON.stringify({
					type: 'stop_simulation',
				})
			)
		}

		// Update processing status
		const statusElement = document.getElementById('processing-status')
		if (statusElement) {
			statusElement.innerHTML =
				'<i class="fas fa-circle text-secondary me-2"></i><span>Ready</span>'
		}
	}
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
	window.dashboard = new SportsAnalyticsDashboard()
})
