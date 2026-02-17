"""
Core Commands Cog

Basic bot management commands.
"""

import logging
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from ..bot import Nurireine

logger = logging.getLogger(__name__)


class CoreCommands(commands.Cog):
    """Core bot commands for channel management."""
    
    def __init__(self, bot: "Nurireine"):
        self.bot = bot
    
    @commands.hybrid_command(
        name='here', 
        description="현재 채널을 봇이 대화를 주시하는 활성 채널로 설정합니다."
    )
    async def set_active_channel(self, ctx: commands.Context) -> None:
        """Set the current channel as the active channel for this guild."""
        if not ctx.guild:
            await ctx.send("이 명령어는 서버에서만 사용할 수 있어요.")
            return
        
        self.bot.active_channels[ctx.guild.id] = ctx.channel.id
        self.bot.db.save_active_channel(
            ctx.guild.id, ctx.channel.id, ctx.channel.name
        )
        
        await ctx.send(f"✅ 이제부터 **#{ctx.channel.name}** 채널의 대화를 귀담아들을게요!")
        logger.info(f"Active channel set for '{ctx.guild.name}': #{ctx.channel.name}")
    
    @commands.hybrid_command(
        name='leave', 
        description="활성 채널 설정을 해제합니다."
    )
    async def remove_active_channel(self, ctx: commands.Context) -> None:
        """Remove the active channel setting for this guild."""
        if not ctx.guild:
            await ctx.send("이 명령어는 서버에서만 사용할 수 있어요.")
            return
        
        if ctx.guild.id not in self.bot.active_channels:
            await ctx.send("설정된 활성 채널이 없습니다.")
            return
        
        del self.bot.active_channels[ctx.guild.id]
        self.bot.db.remove_active_channel(ctx.guild.id)
        
        # Clear message queue for this channel
        self.bot._message_handler.clear_channel(ctx.channel.id)
        
        await ctx.send("💤 이제 대화 감시를 중단하고 쉴게요.")
        logger.info(f"Active channel removed for '{ctx.guild.name}'")
    
    @commands.hybrid_command(
        name='status',
        description="봇의 현재 상태를 확인합니다."
    )
    async def show_status(self, ctx: commands.Context) -> None:
        """Show the bot's current status."""
        from ..health import get_health_checker
        
        health = get_health_checker()
        status = health.get_status()
        
        embed = discord.Embed(
            title="🔧 Nurireine 상태",
            color=discord.Color.green() if status["status"] == "healthy" else discord.Color.orange()
        )
        
        # Overall status
        status_emoji = "✅" if status["status"] == "healthy" else "⚠️"
        embed.add_field(
            name="전체 상태", 
            value=f"{status_emoji} {status['status'].upper()}", 
            inline=True
        )
        
        # Uptime
        uptime_hours = status["uptime_seconds"] / 3600
        embed.add_field(
            name="가동 시간", 
            value=f"{uptime_hours:.1f}시간", 
            inline=True
        )
        
        # AI Systems
        ai_systems = status["ai_systems"]
        ai_status_text = (
            f"로딩: {'✅' if ai_systems['loaded'] else '❌'}\n"
            f"게이트키퍼: {ai_systems['gatekeeper']}\n"
            f"메모리: {ai_systems['memory']}\n"
            f"LLM: {ai_systems['llm']}"
        )
        embed.add_field(name="AI 시스템", value=ai_status_text, inline=True)
        
        # Active channel for this guild
        if ctx.guild and ctx.guild.id in self.bot.active_channels:
            channel_id = self.bot.active_channels[ctx.guild.id]
            channel = ctx.guild.get_channel(channel_id)
            channel_name = f"#{channel.name}" if channel else f"ID: {channel_id}"
            embed.add_field(name="활성 채널", value=channel_name, inline=True)
        else:
            embed.add_field(name="활성 채널", value="없음", inline=True)
        
        # Memory stats
        if self.bot.memory:
            l1_channels = len(self.bot.memory._l1_buffers)
            l2_channels = len(self.bot.memory._l2_summaries)
            embed.add_field(
                name="메모리", 
                value=f"L1: {l1_channels}개 채널\nL2: {l2_channels}개 채널", 
                inline=True
            )
        
        # Statistics
        stats = status["statistics"]
        embed.add_field(
            name="통계",
            value=(
                f"분석: {stats['success_counts']['analyses']}회\n"
                f"응답: {stats['success_counts']['responses']}회\n"
                f"오류: {stats['total_errors']}회"
            ),
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    @commands.hybrid_command(
        name='health',
        description="봇의 간단한 상태 확인 (헬스체크)"
    )
    async def health_check(self, ctx: commands.Context) -> None:
        """Quick health check endpoint."""
        from ..health import get_health_checker
        
        health = get_health_checker()
        is_healthy = health.is_healthy()
        
        status_emoji = "✅" if is_healthy else "⚠️"
        status_text = "정상" if is_healthy else "일부 기능 제한"
        
        embed = discord.Embed(
            title=f"{status_emoji} 헬스 체크",
            description=f"상태: **{status_text}**",
            color=discord.Color.green() if is_healthy else discord.Color.orange()
        )
        
        # Basic component status
        status = health.get_status()
        ai_systems = status["ai_systems"]
        
        components = []
        components.append(f"{'✅' if ai_systems['loaded'] else '❌'} AI 시스템")
        components.append(f"{'✅' if ai_systems['llm'] == 'healthy' else '⚠️'} LLM")
        components.append(f"{'✅' if ai_systems['memory'] == 'healthy' else '⚠️'} 메모리")
        
        embed.add_field(
            name="구성 요소",
            value="\n".join(components),
            inline=False
        )
        
        # Uptime
        uptime_hours = status["uptime_seconds"] / 3600
        embed.add_field(
            name="가동 시간",
            value=f"{uptime_hours:.1f}시간",
            inline=True
        )
        
        # Total operations
        embed.add_field(
            name="처리 완료",
            value=f"{status['statistics']['total_operations']}건",
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    @commands.hybrid_command(
        name='stats',
        description="상세한 성능 및 운영 통계를 확인합니다."
    )
    async def show_stats(self, ctx: commands.Context) -> None:
        """Show detailed performance and operational statistics."""
        from ..metrics import get_metrics_collector
        
        metrics = get_metrics_collector()
        stats_dict = metrics.get_stats_dict()
        
        embed = discord.Embed(
            title="📊 상세 통계",
            description=f"수집 시작: {stats_dict['reset_info']['last_reset'][:19]}",
            color=discord.Color.blue()
        )
        
        # Response metrics
        resp = stats_dict["response_metrics"]
        embed.add_field(
            name="🤖 응답 생성",
            value=(
                f"총 요청: {resp['total']}건\n"
                f"성공: {resp['successful']}건\n"
                f"실패: {resp['failed']}건\n"
                f"실패율: {resp['failure_rate']}\n"
                f"평균 응답 시간: {resp['avg_latency_ms']}ms\n"
                f"P95 응답 시간: {resp['p95_latency_ms']}ms"
            ),
            inline=False
        )
        
        # Retrieval metrics
        retr = stats_dict["retrieval_metrics"]
        embed.add_field(
            name="🧠 메모리 검색",
            value=(
                f"총 검색: {retr['total']}건\n"
                f"히트: {retr['hits']}건\n"
                f"미스: {retr['misses']}건\n"
                f"히트율: {retr['hit_rate']}"
            ),
            inline=True
        )
        
        # Analysis metrics
        anal = stats_dict["analysis_metrics"]
        embed.add_field(
            name="🔍 컨텍스트 분석",
            value=(
                f"총 분석: {anal['total']}건\n"
                f"성공: {anal['successful']}건\n"
                f"실패: {anal['failed']}건"
            ),
            inline=True
        )
        
        # Uptime
        embed.add_field(
            name="⏱️ 가동 시간",
            value=f"{stats_dict['uptime_hours']:.1f}시간",
            inline=True
        )
        
        # Reset info
        next_reset = stats_dict['reset_info']['next_reset'][:19]
        embed.set_footer(text=f"다음 리셋: {next_reset}")
        
        await ctx.send(embed=embed)
    
    @commands.command(name="sync")
    @commands.is_owner()
    async def sync_commands(self, ctx: commands.Context) -> None:
        """(Owner Only) Sync slash commands to Discord."""
        await ctx.bot.tree.sync()
        await ctx.send("✅ Commands synced!")
        logger.info("event=commands_synced user={ctx.author.id}")
    
    @commands.hybrid_command(
        name='testtimer',
        description="최근 AI 대화의 처리 시간 통계를 확인합니다."
    )
    async def show_performance_stats(self, ctx: commands.Context) -> None:
        """Show performance stats for the last interaction."""
        stats = self.bot.last_stats.copy()
        if not stats:
            await ctx.send("아직 기록된 대화 통계가 없어요.")
            return

        embed = discord.Embed(
            title="⏱️ AI 처리 성능 분석",
            description="최근 대화의 단계별 소요 시간입니다.",
            color=discord.Color.magenta()
        )
        
        # 1. Queue Wait (Delay before processing starts)
        if "arrival_wall" in stats and "process_start_wall" in stats:
            queue_time = stats["process_start_wall"] - stats["arrival_wall"]
            embed.add_field(name="1️⃣ 처리 대기 (지연)", value=f"{queue_time*1000:.0f}ms", inline=True)
        else:
            embed.add_field(name="1️⃣ 처리 대기", value="N/A", inline=True)
            
        # 2. Gatekeeper (SLM Analysis)
        slm_total = stats.get("slm_total_duration", 0)
        embed.add_field(name="2️⃣ Context 분석 (전체)", value=f"{slm_total:.2f}s", inline=True)
        
        # 3. LLM Processing (Streaming)
        llm_duration = stats.get("llm_duration", 0)
        embed.add_field(name="3️⃣ LLM 생성 (스트리밍)", value=f"{llm_duration:.2f}s", inline=True)

        # Detailed Breakdown Row
        details = []
        if "slm" in stats: details.append(f"BERT/SLM: {stats['slm']:.2f}s")
        if "l3_search" in stats: details.append(f"기억조회: {stats['l3_search']:.2f}s")
        if "l3_save" in stats: details.append(f"기억저장: {stats['l3_save']:.2f}s")
        
        if details:
            embed.add_field(name="🔍 분석 상세", value=" | ".join(details), inline=False)

        # Total Turnaround
        total = stats.get("total_turnaround", 0)
        embed.add_field(name="⚡ 총 소요 시간", value=f"**{total:.2f}s** (응답 완료까지)", inline=False)
        
        # Footer with timestamps
        if "process_start_wall" in stats:
            from datetime import datetime
            start_dt = datetime.fromtimestamp(stats["process_start_wall"])
            embed.set_footer(text=f"처리 시작 시각: {start_dt.strftime('%H:%M:%S')}")
            
        await ctx.send(embed=embed)

    @commands.command(name="clearmemory")
    @commands.is_owner()
    async def clear_memory(self, ctx: commands.Context) -> None:
        """(Owner Only) Clear L1 memory for the current channel."""
        if not self.bot.memory:
            await ctx.send("메모리 시스템이 초기화되지 않았습니다.")
            return
        
        self.bot.memory.clear_l1_buffer(ctx.channel.id)
        await ctx.send("✅ 이 채널의 L1 메모리를 초기화했습니다.")
        logger.info(f"L1 memory cleared for channel {ctx.channel.id}")


async def setup(bot: "Nurireine") -> None:
    """Setup function for loading the cog."""
    await bot.add_cog(CoreCommands(bot))
